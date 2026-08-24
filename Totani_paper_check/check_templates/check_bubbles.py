import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from totani_helpers.totani_io import (
    read_counts_and_ebounds, lonlat_grids, pixel_solid_angle_map,
    read_exposure, resample_exposure_logE, load_mask_any_shape,
)
from totani_helpers.bubbles_templates import nnls_fit_per_energy, build_model_counts

repo_dir = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
data_dir = os.path.join(repo_dir, "fermi_data", "totani")
tmpl_dir = os.path.join(data_dir, "processed", "templates")
proc_dir = os.path.join(data_dir, "processed")

SPLIT_MODE = os.environ.get("BUBBLES_SPLIT_MODE", "closure").strip().lower()

k2 = 2
binsz = 0.125
roi_lon = 60.0
roi_lat = 60.0
disk_cut_deg = 0.0

BG_COMPONENTS = [
    "gas", "ics", "iso", "ps", "loopA", "loopB",
    "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
    "fb_flat",
]

# ── 0. Load data + geometry ──────────────────────────────────────────────────
counts_path = os.path.join(proc_dir, "counts_ccube_1000to1000000.fits")
counts, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(counts_path)
nE, ny, nx = counts.shape
Egev = np.asarray(Ectr_mev) / 1000.0

expo_raw, E_expo = read_exposure(os.path.join(proc_dir, "expcube_1000to1000000.fits"))
expo = resample_exposure_logE(expo_raw, E_expo, Ectr_mev)

wcs = WCS(hdr).celestial
omega = pixel_solid_angle_map(wcs, ny, nx, binsz)
lon2d, lat2d = lonlat_grids(wcs, ny, nx)
roi2d = (np.abs(lon2d) <= roi_lon) & (np.abs(lat2d) <= roi_lat)

data_ok2d = np.any(np.isfinite(counts) & np.isfinite(expo) & (expo > 0), axis=0)

# Totani region: full ROI excluding disk, no source mask, no boundary
disk_cut = np.abs(lat2d) >= disk_cut_deg
totani_region = roi2d & disk_cut & data_ok2d
mfit_totani = np.asarray(totani_region, bool)
fit3d_totani = np.broadcast_to(mfit_totani[None, :, :], (nE, ny, nx)).copy()

# ── 1. Load templates ────────────────────────────────────────────────────────
flat = fits.getdata(os.path.join(tmpl_dir, "mu_fb_flat_counts.fits")).astype(float)
pos  = fits.getdata(os.path.join(tmpl_dir, "mu_fb_pos_counts.fits")).astype(float)
neg  = fits.getdata(os.path.join(tmpl_dir, "mu_fb_neg_counts.fits")).astype(float)

print("=== Raw value check ===")
print(f"flat: min={flat.min():.3e}  max={flat.max():.3e}  sum={flat.sum():.3e}")
print(f"pos:  min={pos.min():.3e}   max={pos.max():.3e}   sum={pos.sum():.3e}")
print(f"neg:  min={neg.min():.3e}   max={neg.max():.3e}   sum={neg.sum():.3e}")
print(f"neg.min() <= 0: {neg.min() <= 0}  (should be True)")
print(f"split mode (env BUBBLES_SPLIT_MODE): {SPLIT_MODE}")

# ── 2. Spatial sum images ────────────────────────────────────────────────────
sigma_pix_disp = 1.0 / binsz

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)

for j, (cube, title) in enumerate(zip([flat, pos, neg], ["fb_flat", "fb_pos", "fb_neg"])):
    img = np.sum(cube, axis=0)
    img_sm = gaussian_filter(img, sigma=sigma_pix_disp, mode="constant", cval=0.0)

    vmax = np.nanpercentile(np.abs(img_sm), 99)
    im = axes[0, j].imshow(img_sm, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, j].set_title(f"{title} (summed, 1deg smoothed)")
    fig.colorbar(im, ax=axes[0, j], fraction=0.046)

    if title == "fb_pos":
        img_disp = np.clip(img_sm, 0.0, None)
        cmap = "magma"
        vmin, vmax2 = 0.0, np.nanpercentile(img_disp, 99)
    elif title == "fb_neg":
        img_disp = np.clip(img_sm, None, 0.0)
        cmap = "viridis"
        vmin, vmax2 = np.nanpercentile(img_disp, 1), 0.0
    else:
        img_disp = img_sm
        cmap = "RdBu_r"
        vmax2 = np.nanpercentile(np.abs(img_disp), 99)
        vmin, vmax2 = -vmax2, vmax2

    im2 = axes[1, j].imshow(img_disp, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax2)
    axes[1, j].set_title(f"{title} (display-clipped)")
    fig.colorbar(im2, ax=axes[1, j], fraction=0.046)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("plots_check_bubbles/bubble_templates_summed.png", dpi=150)
print("Saved bubble_templates_summed.png")

# ── 3. Spectrum check ────────────────────────────────────────────────────────
flat_spec = flat.sum(axis=(1, 2))
pos_spec  = pos.sum(axis=(1, 2))
neg_spec  = neg.sum(axis=(1, 2))

print("\n=== Spectra ===")
print(f"flat_spec all positive: {np.all(flat_spec > 0)}")
print(f"pos_spec  all positive: {np.all(pos_spec  > 0)}")
print(f"neg_spec  all negative: {np.all(neg_spec  < 0)}  (should be True)")
print(f"neg_spec  all positive: {np.all(neg_spec  > 0)}  (should be False)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1.loglog(Egev, flat_spec, label="fb_flat", lw=2, color="C0")
ax1.loglog(Egev, np.abs(pos_spec), label="fb_pos", lw=2, ls="--", color="C1")
ax1.set_ylabel("counts (summed over pixels)")
ax1.legend()
ax1.set_title("fb_pos should trace fb_flat closely")
ax1.grid(True, alpha=0.3)

neg_plot = -neg_spec if np.median(neg_spec) < 0 else neg_spec
sign_label = "-fb_neg (negated for display)" if np.median(neg_spec) < 0 else "fb_neg (WARNING: already positive!)"
ax2.loglog(Egev, neg_plot, label=sign_label, lw=2, color="C2")
ax2.set_xlabel("E [GeV]")
ax2.set_ylabel("|counts|")
ax2.legend()
ax2.set_title("fb_neg |spectrum| should resemble gas (rising to low E)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots_check_bubbles/bubble_spectra_check.png", dpi=150)
print("Saved bubble_spectra_check.png")

# ── 4. Mask integrity checks ─────────────────────────────────────────────────
pos_mask  = fits.getdata(os.path.join(tmpl_dir, "fb_pos_mask.fits")).astype(bool)
neg_mask  = fits.getdata(os.path.join(tmpl_dir, "fb_neg_mask.fits")).astype(bool)
flat_mask = fits.getdata(os.path.join(tmpl_dir, "fb_flat_mask.fits")).astype(bool)

if pos_mask.ndim == 3:  pos_mask  = pos_mask[0]
if neg_mask.ndim == 3:  neg_mask  = neg_mask[0]
if flat_mask.ndim == 3: flat_mask = flat_mask[0]

overlap = int(np.sum(pos_mask & neg_mask))
outside = int(np.sum((pos_mask | neg_mask) & ~flat_mask))
print("\n=== Mask checks ===")
print(f"pos pixels:                            {pos_mask.sum()}")
print(f"neg pixels:                            {neg_mask.sum()}")
print(f"flat pixels:                           {flat_mask.sum()}")
print(f"overlap pixels (must be 0):            {overlap}")
print(f"pos/neg outside flat boundary (= 0):   {outside}")
print(f"pos/flat pixel ratio (expect 0.7-0.9): {pos_mask.sum()/flat_mask.sum():.2f}")

# ── 5. lon/lat overlay ───────────────────────────────────────────────────────
hdr_mask = fits.getheader(os.path.join(tmpl_dir, "fb_flat_mask.fits"))
wcs2d = WCS(hdr_mask).celestial

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection=wcs2d)
img = pos_mask.astype(float) - neg_mask.astype(float)
ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Galactic Longitude")
ax.set_ylabel("Galactic Latitude")
ax.coords[0].set_format_unit("deg")
ax.coords[1].set_format_unit("deg")
ax.set_title("pos (red) / neg (blue)\nneg should rim bubble edges near disk")
plt.tight_layout()
plt.savefig("plots_check_bubbles/bubble_posneg_lonlat.png", dpi=150)
print("Saved bubble_posneg_lonlat.png")

# ── 6. Closure check — uses totani_region to match build script ──────────────
print(f"\n=== Closure check at k={k2} (E={Egev[k2]:.2f} GeV) ===")

bg_templates = []
for name in BG_COMPONENTS:
    path = os.path.join(tmpl_dir, f"mu_{name}_counts.fits")
    if not os.path.exists(path):
        print(f"  WARNING: missing {path}, skipping")
        continue
    t = fits.getdata(path).astype(float)
    bg_templates.append(t)
    print(f"  loaded {name}: sum[k2]={t[k2].sum():.3e}")

print(f"  Fitting {len(bg_templates)} background templates...")
coeffs_bg = nnls_fit_per_energy(
    counts=counts,
    templates=bg_templates,
    mask3d=fit3d_totani,   # use totani_region
    ridge=0.0,
)
print(f"  Background coefficients at k={k2}: {coeffs_bg[k2]}")

model_bg = build_model_counts(coeffs=coeffs_bg, templates=bg_templates)
resid_at_k2 = counts[k2] - model_bg[k2]
bubble_contribution = pos[k2] + neg[k2]
residual_after = resid_at_k2 - bubble_contribution

print(f"\n  data sum at k2:               {float(counts[k2][mfit_totani].sum()):.6e}")
print(f"  background model sum at k2:   {float(model_bg[k2][mfit_totani].sum()):.6e}")
print(f"  raw residual sum at k2:       {float(resid_at_k2[mfit_totani].sum()):.6e}")
print(f"  mu_pos sum at k2:             {float(pos[k2][mfit_totani].sum()):.6e}")
print(f"  mu_neg sum at k2:             {float(neg[k2][mfit_totani].sum()):.6e}  (should be <= 0)")
print(f"  mu_pos + mu_neg sum at k2:    {float(bubble_contribution[mfit_totani].sum()):.6e}")
if SPLIT_MODE == "closure":
    print(f"  residual AFTER bubbles at k2: {float(residual_after[mfit_totani].sum()):.6e}  <-- should be ~0")
    print(f"  max |residual| after bubbles: {float(np.abs(residual_after[mfit_totani]).max()):.6e}  <-- should be ~0")
else:
    print(f"  residual AFTER bubbles at k2: {float(residual_after[mfit_totani].sum()):.6e}")
    print(f"  max |residual| after bubbles: {float(np.abs(residual_after[mfit_totani]).max()):.6e}")

k5 = 5
resid_at_k5 = counts[k5] - model_bg[k5]
bubble_contribution = pos[k5] + neg[k5]
residual_after = resid_at_k5 - bubble_contribution

print(f"\n  data sum at k5:               {float(counts[k5][mfit_totani].sum()):.6e}")
print(f"  background model sum at k5:   {float(model_bg[k5][mfit_totani].sum()):.6e}")
print(f"  raw residual sum at k5:       {float(resid_at_k5[mfit_totani].sum()):.6e}")
print(f"  mu_pos sum at k5:             {float(pos[k5][mfit_totani].sum()):.6e}")
print(f"  mu_neg sum at k5:             {float(neg[k5][mfit_totani].sum()):.6e}  (should be <= 0)")
print(f"  mu_pos + mu_neg sum at k5:    {float(bubble_contribution[mfit_totani].sum()):.6e}")
print(f"  residual AFTER bubbles at k5: {float(residual_after[mfit_totani].sum()):.6e}  <-- should NOT be ~0")
print(f"  max |residual| after bubbles: {float(np.abs(residual_after[mfit_totani]).max()):.6e}  <-- should NOT be ~0")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

resid_at_k2_sm = gaussian_filter(resid_at_k2, sigma=sigma_pix_disp, mode="constant", cval=0.0)
residual_after_sm = gaussian_filter(residual_after, sigma=sigma_pix_disp, mode="constant", cval=0.0)
vmax = float(np.nanpercentile(np.abs(resid_at_k2_sm[mfit_totani]), 99))
for ax, img, title in zip(axes,
    [resid_at_k2_sm, residual_after_sm],
    [
        f"Residual before bubbles (k={k2}) (1deg smoothed)",
        (
            f"Residual after pos+neg (k={k2}) (1deg smoothed) — should be ~0"
            if SPLIT_MODE == "closure"
            else f"Residual after pos+neg (k={k2}) (1deg smoothed)"
        ),
    ]):
    disp = np.where(mfit_totani, img, np.nan)
    im = ax.imshow(disp, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("plots_check_bubbles/bubble_closure_check.png", dpi=150)
print("\nSaved bubble_closure_check.png")
if SPLIT_MODE == "closure":
    print("The right panel should be uniformly ~0 (white) if closure is correct.")
else:
    print("In totani split mode, closure is not expected; right panel will not be ~0.")

# ── 6b. Replicated construction fit — must use same mask as build script ─────
print(f"\n=== Replicated construction fit at k={k2} only ===")

bg_templates_k2 = []
bg_names_used = []
for name in BG_COMPONENTS:
    path = os.path.join(tmpl_dir, f"mu_{name}_counts.fits")
    if not os.path.exists(path):
        continue
    t = fits.getdata(path).astype(float)[k2]
    bg_templates_k2.append(t)
    bg_names_used.append(name)

y_k2 = counts[k2][mfit_totani].ravel()
X_k2 = np.vstack([t[mfit_totani].ravel() for t in bg_templates_k2]).T
good = np.isfinite(y_k2) & np.all(np.isfinite(X_k2), axis=1)
coeffs_k2, _ = nnls(X_k2[good], y_k2[good])

print("  Single-bin NNLS coefficients at k2 (totani_region):")
for name, c in zip(bg_names_used, coeffs_k2):
    print(f"    {name}: {c:.6e}")

model_k2 = sum(c * t for c, t in zip(coeffs_k2, bg_templates_k2))
resid_k2 = np.where(mfit_totani, counts[k2] - model_k2, 0.0)

sigma_pix = 1.0 / binsz
resid_sm = gaussian_filter(resid_k2, sigma=sigma_pix, mode="constant", cval=0.0)

pos_sel = mfit_totani & np.isfinite(resid_sm) & (resid_sm > 0)
neg_sel = mfit_totani & np.isfinite(resid_sm) & (resid_sm < 0)
unassigned = mfit_totani & ~(pos_sel | neg_sel)
pos_sel |= unassigned & np.isfinite(resid_k2) & (resid_k2 >= 0)
neg_sel |= unassigned & np.isfinite(resid_k2) & (resid_k2 < 0)

if SPLIT_MODE == "totani":
    mu_pos_k2 = np.where(pos_sel, np.clip(resid_sm, 0.0, None), 0.0)
    mu_neg_k2 = np.where(neg_sel, np.clip(resid_sm, None, 0.0), 0.0)
else:
    mu_pos_k2 = np.where(pos_sel, resid_k2, 0.0)
    mu_neg_k2 = np.where(neg_sel, resid_k2, 0.0)

total_k2 = model_k2 + mu_pos_k2 + mu_neg_k2
closure_resid = np.where(mfit_totani, counts[k2] - total_k2, 0.0)

print(f"\n  data sum:             {float(counts[k2][mfit_totani].sum()):.6e}")
print(f"  bg model sum:         {float(model_k2[mfit_totani].sum()):.6e}")
print(f"  raw residual sum:     {float(resid_k2[mfit_totani].sum()):.6e}")
print(f"  mu_pos_k2 sum:        {float(mu_pos_k2[mfit_totani].sum()):.6e}")
print(f"  mu_neg_k2 sum:        {float(mu_neg_k2[mfit_totani].sum()):.6e}")
if SPLIT_MODE == "closure":
    print(f"  closure residual sum: {float(closure_resid[mfit_totani].sum()):.6e}  <-- must be ~0")
    print(f"  max |closure resid|:  {float(np.abs(closure_resid[mfit_totani]).max()):.6e}  <-- must be ~0")
else:
    print(f"  closure residual sum: {float(closure_resid[mfit_totani].sum()):.6e}")
    print(f"  max |closure resid|:  {float(np.abs(closure_resid[mfit_totani]).max()):.6e}")

print(f"\n  Saved mu_pos[k2] sum: {float(pos[k2][mfit_totani].sum()):.6e}")
print(f"  Constructed pos sum:  {float(mu_pos_k2[mfit_totani].sum()):.6e}")
print(f"  Difference:           {float((pos[k2]-mu_pos_k2)[mfit_totani].sum()):.6e}  <-- should be ~0")

print(f"\n  Saved mu_neg[k2] sum: {float(neg[k2][mfit_totani].sum()):.6e}")
print(f"  Constructed neg sum:  {float(mu_neg_k2[mfit_totani].sum()):.6e}")
print(f"  Difference:           {float((neg[k2]-mu_neg_k2)[mfit_totani].sum()):.6e}  <-- should be ~0")
