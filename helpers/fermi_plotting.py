import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.image as mpimg

# ── Conference-presentation font sizes ──────────────────────────────────────
CONF_TITLE_FS   = 30   # panel / figure titles
CONF_LABEL_FS   = 28   # axis labels
CONF_TICK_FS    = 26   # tick labels
CONF_LEGEND_FS  = 26   # legend entries
CONF_ANNOT_FS   = 24   # in-plot annotations
CONF_HEADER_FS  = 26   # suptitle / header text


def latex_sci(x, sig=2):
    x = float(x)
    if x == 0.0:
        return r"0"
    s = f"{x:.{int(sig)}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp_i = int(exp)
    return rf"{mant}\times10^{{{exp_i}}}"


def operator_title(op):
    op = str(op)
    titles = {
        "rayleigh_even":   r"Rayleigh (even, $\mathcal{O} \sim F^2$)",
        "rayleigh_odd":    r"Rayleigh (odd, $\mathcal{O} \sim F\tilde{F}$)",
        "rayleigh_full":   r"Rayleigh (full)",
        "dipole_magnetic": r"Magnetic dipole ($d=5$)",
        "dipole_electric": r"Electric dipole ($d=5$)",
        "charge_radius":   r"Charge radius ($d=6$)",
        "anapole":         r"Anapole ($d=6$)",
    }
    return titles.get(op, op)


def _theme_colors():
    """Return text/line colors consistent with the current rcParams theme."""
    fc = mpl.rcParams.get("figure.facecolor", "white")
    try:
        r, g, b = mpl.colors.to_rgb(fc)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        is_dark = lum < 0.4 or fc == "none"
    except Exception:
        is_dark = False
    text_col   = "white" if is_dark else "black"
    thresh_col = "white" if is_dark else "black"
    return text_col, thresh_col


def add_hatched_region_from_contour(
    *,
    ax,
    X,
    Y,
    Z,
    level,
    upper_level,
    hatch="////",
    edgecolor="c",
    zorder=3,
    outline_lw=1.5,
):
    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=[float(level), float(upper_level)],
        colors=["none"],
        antialiased=False,
    )

    segs = []
    if hasattr(cf, "allsegs") and isinstance(cf.allsegs, (list, tuple)) and len(cf.allsegs) > 0:
        segs = cf.allsegs[0]

    for col in getattr(cf, "collections", []):
        try:
            col.remove()
        except Exception:
            pass

    if not segs:
        return False

    for seg in segs:
        seg = np.asarray(seg, dtype=float)
        if seg.ndim != 2 or seg.shape[0] < 3:
            continue
        if not np.allclose(seg[0], seg[-1]):
            seg = np.vstack([seg, seg[0]])
        codes = np.full(seg.shape[0], Path.LINETO, dtype=int)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(seg, codes)

        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=edgecolor,
            hatch=hatch,
            lw=outline_lw,
            zorder=zorder,
        )

        with mpl.rc_context({"hatch.color": edgecolor, "hatch.linewidth": 1.0}):
            ax.add_patch(patch)

    return True


def _style_ax_for_conference(ax, *, xlabel=None, ylabel=None, title=None):
    """Apply conference-scale fonts to a single Axes."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=CONF_LABEL_FS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=CONF_LABEL_FS)
    if title is not None:
        ax.set_title(title, fontsize=CONF_TITLE_FS)
    ax.tick_params(labelsize=CONF_TICK_FS)
    if ax.get_legend() is not None:
        plt.setp(ax.get_legend().get_texts(), fontsize=CONF_LEGEND_FS)


def make_combined_tau_vs_lambda_beamer(
    *,
    operators,
    compute_curve,
    tau_needed,
    tau_energy_label,
    out_base,
    header_text,
    ncols=3,
    fermion_type=None,
):
    """
    Multi-panel τ_max vs Λ figure, one panel per operator.

    Parameters
    ----------
    operators        : list[str]
    compute_curve    : callable(op) → (Lambda_grid, tau_max_lambda)
    tau_needed       : float
    tau_energy_label : str
    out_base         : str   path without extension
    header_text      : str   suptitle
    ncols            : int
    fermion_type     : str or None  (used in suptitle if provided)
    """
    operators = [str(o) for o in operators]
    nops = int(len(operators))
    if nops <= 0:
        raise ValueError("operators must be a non-empty list")

    ncols = int(max(1, ncols))
    nrows = int(np.ceil(float(nops) / float(ncols)))

    text_col, thresh_col = _theme_colors()

    # Taller panels for conference readability
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * float(ncols), 4.5 * float(nrows)),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    suptitle = str(header_text)
    if fermion_type is not None:
        ft_label = "Dirac" if str(fermion_type) == "dirac" else "Majorana"
        suptitle = rf"{ft_label} DM — " + suptitle
    fig.suptitle(suptitle, fontsize=CONF_HEADER_FS, y=0.985, color=text_col)

    tau_fermi = 1e-2
    tau_cta   = 1e-3

    for i, op in enumerate(operators):
        r = int(i // ncols)
        c = int(i % ncols)
        ax = axes[r, c]

        Lambda_grid, tau_max_lambda = compute_curve(op)
        Lambda_grid    = np.asarray(Lambda_grid,    dtype=float)
        tau_max_lambda = np.asarray(tau_max_lambda, dtype=float)

        ax.plot(Lambda_grid, tau_max_lambda, lw=2.5, label=r"$\tau_{\max}(\Lambda)$")

        ax.axhline(float(tau_needed), color=thresh_col, ls="--", lw=2.0,
                   label=r"$\tau_{\rm needed}$")
        ax.axhline(tau_fermi, color="gold",   ls="-",  lw=1.8, alpha=0.9,
                   label=rf"Fermi-LAT ($\tau={tau_fermi:.0e}$)")
        ax.axhline(tau_cta,   color="lime",   ls="-",  lw=1.8, alpha=0.9,
                   label=rf"CTA ($\tau={tau_cta:.0e}$)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(operator_title(op), fontsize=CONF_TITLE_FS - 1, color=text_col)
        ax.tick_params(labelsize=CONF_TICK_FS)

        # Shade region above Fermi sensitivity
        tau_arr = np.asarray(tau_max_lambda, dtype=float)
        above_fermi = tau_arr >= tau_fermi
        if np.any(above_fermi):
            ax.fill_between(Lambda_grid, tau_arr, tau_fermi,
                            where=above_fermi,
                            alpha=0.15, color="gold", zorder=1,
                            label="Fermi-sensitive region")

        if r == (nrows - 1):
            ax.set_xlabel(r"$\Lambda\,[\mathrm{GeV}]$", fontsize=CONF_LABEL_FS)
        if c == 0:
            ax.set_ylabel(
                rf"$\tau_{{\max}}$ ({str(tau_energy_label)})",
                fontsize=CONF_LABEL_FS,
            )

        ax.legend(fontsize=CONF_LEGEND_FS - 1, frameon=True,
                  framealpha=0.3, edgecolor=text_col, loc="lower left")

    for j in range(nops, nrows * ncols):
        r = int(j // ncols)
        c = int(j % ncols)
        axes[r, c].axis("off")

    fig.subplots_adjust(top=0.88, hspace=0.38, wspace=0.28)

    fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf",           bbox_inches="tight")
    plt.close(fig)


def make_combined_tau_grid_png_beamer(
    *,
    operators,
    png_paths,
    out_base,
    header_text,
    ncols=3,
    fermion_type=None,
):
    """
    Assemble individual tau-grid PNGs into a single multi-panel figure
    suitable for conference beamer slides.

    Parameters
    ----------
    operators    : list[str]
    png_paths    : list[str]   one PNG path per operator (same order)
    out_base     : str         path without extension
    header_text  : str         suptitle
    ncols        : int
    fermion_type : str or None
    """
    operators = [str(o) for o in operators]
    png_paths = [str(p) for p in png_paths]
    if len(operators) != len(png_paths):
        raise ValueError("operators and png_paths must have the same length")

    nops = int(len(operators))
    if nops <= 0:
        raise ValueError("operators must be a non-empty list")

    ncols = int(max(1, ncols))
    nrows = int(np.ceil(float(nops) / float(ncols)))

    text_col, _ = _theme_colors()

    # Larger figure to avoid label crowding on slides
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * float(ncols), 4.8 * float(nrows)),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    suptitle = str(header_text)
    if fermion_type is not None:
        ft_label = "Dirac" if str(fermion_type) == "dirac" else "Majorana"
        suptitle = rf"{ft_label} DM — " + suptitle
    fig.text(
        0.5,
        0.975,
        suptitle,
        ha="center",
        va="top",
        fontsize=CONF_HEADER_FS,
        color=text_col,
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=0.06,
        top=0.91,
        hspace=0.18,
        wspace=0.08,
    )

    for i, (op, p) in enumerate(zip(operators, png_paths)):
        r = int(i // ncols)
        c = int(i % ncols)
        ax = axes[r, c]

        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(operator_title(op), fontsize=CONF_TITLE_FS - 1,
                     pad=8, color=text_col)
        ax.axis("off")

    for j in range(nops, nrows * ncols):
        r = int(j // ncols)
        c = int(j % ncols)
        axes[r, c].axis("off")

    fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf",           bbox_inches="tight")
    plt.close(fig)
