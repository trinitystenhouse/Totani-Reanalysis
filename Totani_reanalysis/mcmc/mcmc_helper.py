from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np


@dataclass
class MCMCResult:
    labels: List[str]
    f_ml: np.ndarray
    f_p16: np.ndarray
    f_p50: np.ndarray
    f_p84: np.ndarray
    loglike_ml: float
    chain: np.ndarray
    logprob: np.ndarray
    acceptance_fraction: np.ndarray
    used_emcee: bool


@dataclass
class _LogProbCallable:
    Cobs: np.ndarray
    mu: np.ndarray
    labels_l: List[str]
    bounds: Sequence[Tuple[float, float]]
    f_init: np.ndarray
    iso_prior_sigma_dex: Optional[float]
    iso_prior_mode: str
    iso_prior_center: Optional[float]
    nonstable_prior_sigma_dex: Optional[float]
    nonstable_prior_centers: Optional[dict]
    component_priors: Optional[dict]
    poisson_negative_mode: str
    poisson_cexp_floor: float
    stat: str
    gaussian_flux_reg_counts: float

    def _log10_ratio_gauss(self, x: float, x0: float, sigma_dex: float) -> float:
        if (sigma_dex is None) or (sigma_dex <= 0) or (not np.isfinite(sigma_dex)):
            return 0.0
        if (not np.isfinite(x)) or (not np.isfinite(x0)) or (x <= 0) or (x0 <= 0):
            return -np.inf
        z = np.log10(x / x0)
        return -0.5 * (z / float(sigma_dex)) ** 2

    def _log10_ratio_gauss_upper_only(self, x: float, x0: float, sigma_dex: float) -> float:
        if (sigma_dex is None) or (sigma_dex <= 0) or (not np.isfinite(sigma_dex)):
            return 0.0
        if (not np.isfinite(x)) or (not np.isfinite(x0)) or (x <= 0) or (x0 <= 0):
            return -np.inf
        if x <= x0:
            return 0.0
        z = np.log10(x / x0)
        return -0.5 * (z / float(sigma_dex)) ** 2

    def _log10_ratio_gauss_lower_only(self, x: float, x0: float, sigma_dex: float) -> float:
        if (sigma_dex is None) or (sigma_dex <= 0) or (not np.isfinite(sigma_dex)):
            return 0.0
        if (not np.isfinite(x)) or (not np.isfinite(x0)) or (x <= 0) or (x0 <= 0):
            return -np.inf
        if x >= x0:
            return 0.0
        z = np.log10(x / x0)
        return -0.5 * (z / float(sigma_dex)) ** 2

    def _log10_ratio_gauss_mode(self, x: float, x0: float, sigma_dex: float, mode: str) -> float:
        m = str(mode).strip().lower()
        if m in ("f_upper", "upper", "upper_only"):
            return self._log10_ratio_gauss_upper_only(x, x0, sigma_dex)
        if m in ("f_lower", "lower", "lower_only"):
            return self._log10_ratio_gauss_lower_only(x, x0, sigma_dex)
        return self._log10_ratio_gauss(x, x0, sigma_dex)

    def logprior_gaussian(self, f: np.ndarray) -> float:
        lp = 0.0

        if self.iso_prior_sigma_dex is not None:
            jiso = None
            for key in ("iso", "isotropic"):
                if key in self.labels_l:
                    jiso = self.labels_l.index(key)
                    break

            if jiso is not None:
                center = float(self.iso_prior_center) if (self.iso_prior_center is not None) else float(self.f_init[jiso])
                mode = str(self.iso_prior_mode).strip().lower()
                if mode in ("f_upper", "upper", "upper_only"):
                    lp_i = self._log10_ratio_gauss_upper_only(float(f[jiso]), center, float(self.iso_prior_sigma_dex))
                elif mode in ("f_lower", "lower", "lower_only"):
                    lp_i = self._log10_ratio_gauss_lower_only(float(f[jiso]), center, float(self.iso_prior_sigma_dex))
                else:
                    lp_i = self._log10_ratio_gauss(float(f[jiso]), center, float(self.iso_prior_sigma_dex))
                if not np.isfinite(lp_i):
                    return -np.inf
                lp += lp_i

        if self.nonstable_prior_sigma_dex is not None:
            centers = self.nonstable_prior_centers or {}
            for lab_key, ctr in centers.items():
                lk = str(lab_key).lower()
                if lk in ("iso", "isotropic"):
                    continue
                if lk not in self.labels_l:
                    continue
                j = self.labels_l.index(lk)
                lp_j = self._log10_ratio_gauss(float(f[j]), float(ctr), float(self.nonstable_prior_sigma_dex))
                if not np.isfinite(lp_j):
                    return -np.inf
                lp += lp_j

        if self.component_priors is not None:
            if not isinstance(self.component_priors, dict):
                return -np.inf
            for lab_key, spec in self.component_priors.items():
                lk = str(lab_key).lower()
                if lk not in self.labels_l:
                    continue
                if spec is None:
                    continue
                if not isinstance(spec, dict):
                    return -np.inf

                sigma = spec.get("sigma_dex", None)
                if sigma is None:
                    continue
                sigma = float(sigma)
                if (not np.isfinite(sigma)) or (sigma <= 0):
                    continue

                if "center" not in spec:
                    continue
                center = float(spec.get("center"))
                mode = spec.get("mode", "f")

                j = self.labels_l.index(lk)
                lp_j = self._log10_ratio_gauss_mode(float(f[j]), center, sigma, str(mode))
                if not np.isfinite(lp_j):
                    return -np.inf
                lp += lp_j

        return float(lp)

    def __call__(self, f: np.ndarray) -> float:
        lp = logprior_bounds_inclusive(f, self.bounds)
        if not np.isfinite(lp):
            return -np.inf

        lp_g = self.logprior_gaussian(f)
        if not np.isfinite(lp_g):
            return -np.inf

        with np.errstate(over="ignore", invalid="ignore"):
            Cexp = f @ self.mu
        if np.any(~np.isfinite(Cexp)):
            return -np.inf

        if self.stat == "gaussian_flux":
            ll = loglike_gaussian_flux(
                self.Cobs,
                Cexp,
                reg_counts=float(self.gaussian_flux_reg_counts),
            )
        else:
            ll = loglike_poisson_counts(
                self.Cobs,
                Cexp,
                negative_mode=str(self.poisson_negative_mode),
                cexp_floor=float(self.poisson_cexp_floor),
            )
        if not np.isfinite(ll):
            return -np.inf

        return float(lp + lp_g + ll)


def loglike_poisson_counts(
    Cobs: np.ndarray,
    Cexp: np.ndarray,
    *,
    negative_mode: str = "strict",
    cexp_floor: float = 1e-30,
) -> float:
    """
    Poisson log-likelihood up to an additive constant:
      lnL = sum_i [ Cobs_i * ln(Cexp_i) - Cexp_i ]
    Robust handling of Cexp==0:
      - If Cobs>0 and Cexp==0 -> -inf
      - If Cobs==0 and Cexp==0 -> contributes 0
    In strict/reject mode, negative Cexp is rejected only where Cobs>0.
    Zero-count pixels with Cexp<=0 are treated as empty-support pixels and
    contribute 0 rather than killing the walker.
    """
    Cobs = np.asarray(Cobs, float)
    Cexp = np.asarray(Cexp, float)

    if np.any(~np.isfinite(Cexp)):
        return -np.inf

    mode = str(negative_mode).strip().lower()
    if np.any(Cexp < 0.0):
        if mode in ("strict", "reject"):
            if np.any((Cexp < 0.0) & (Cobs > 0.0)):
                return -np.inf
        elif mode in ("clip", "floor"):
            floor = float(cexp_floor)
            if (not np.isfinite(floor)) or (floor <= 0):
                floor = 1e-30
            Cexp = np.maximum(Cexp, floor)
        else:
            raise ValueError(f"Unknown negative_mode={negative_mode!r}")

    pos = Cexp > 0.0
    if np.any((~pos) & (Cobs > 0.0)):
        return -np.inf

    # Only pos entries contribute Cobs*ln(Cexp); zeros contribute -0 = 0
    return float(np.sum(Cobs[pos] * np.log(Cexp[pos]) - Cexp[pos]))


def loglike_gaussian_flux(
    Cobs: np.ndarray,
    Cexp: np.ndarray,
    *,
    reg_counts: float = 1.0,
) -> float:
    """
    Gaussian log-likelihood expressed in flux space (Pearson chi-squared on counts):
      lnL = -0.5 * sum_i [ (Cobs_i - Cexp_i)^2 / max(Cobs_i, reg_counts) ]

    Derivation: in flux space F = C/D, sigma_F = sqrt(C_obs)/D (Poisson noise).
      chi2_flux = sum [ (F_obs - F_pred)^2 / sigma_F^2 ]
                = sum [ (C_obs - C_exp)^2 / C_obs ]
    The denom D cancels. reg_counts regularises empty cells (default 1 count).
    """
    Cobs = np.asarray(Cobs, float)
    Cexp = np.asarray(Cexp, float)

    if np.any(~np.isfinite(Cexp)):
        return -np.inf
    if np.any(~np.isfinite(Cobs)):
        return -np.inf

    sigma2 = np.maximum(Cobs, float(reg_counts))
    chi2 = np.sum((Cobs - Cexp) ** 2 / sigma2)
    return float(-0.5 * chi2)


def logprior_bounds_inclusive(f: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> float:
    """
    Flat prior within bounds, inclusive:
      lo <= f <= hi
    """
    for x, (lo, hi) in zip(f, bounds):
        if not np.isfinite(x):
            return -np.inf
        if (x < lo) or (x > hi):
            return -np.inf
    return 0.0


def totani_bounds(
    labels: Sequence[str],
    *,
    negative_keys: Sequence[str] = ("nfw", "fb_neg"),
    halo_keys: Optional[Sequence[str]] = None,
    lo_known: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Totani text: known components constrained f>=0 (or >0), some components allowed negative.
    We use inclusive bounds (>=0) to allow "start at 0" for some components.
    """
    keys = list(negative_keys)
    if halo_keys is not None:
        keys.extend(list(halo_keys))
    nk = tuple(str(k).lower() for k in keys)
    bnds: List[Tuple[float, float]] = []
    for lab in labels:
        key = lab.lower()
        if any(n in key for n in nk):
            bnds.append((-np.inf, np.inf))
        else:
            bnds.append((lo_known, np.inf))
    return bnds


def infer_iso_E2dNdE_at_f1(mu_iso: np.ndarray, denom: np.ndarray, Ectr_mev: float) -> float:
    """
    Given iso counts template mu_iso (for f_iso=1) and denom=expo*omega*dE,
    infer the median isotropic E^2 dN/dE that f_iso=1 corresponds to.
    """
    mu_iso = np.asarray(mu_iso, float)
    denom = np.asarray(denom, float)
    good = np.isfinite(mu_iso) & np.isfinite(denom) & (denom > 0)
    if not np.any(good):
        raise RuntimeError("Cannot infer isotropic normalization: denom has no valid entries.")
    I_med = float(np.median(mu_iso[good] / denom[good]))  # dN/dE
    return float((Ectr_mev**2) * I_med)


def totani_init_from_mu_counts(
    labels: Sequence[str],
    *,
    mu: np.ndarray,              # (ncomp, npix) counts-space templates for THIS energy bin
    denom: np.ndarray,           # (npix,) expo*omega*dE on the same masked pixels
    Ectr_mev: float,
    iso_target_E2: Optional[float] = 1e-4, # If None, iso starts at f=1 like other physical templates
    ps_keys: Sequence[str] = ("ps", "point_sources", "pointsources"),
    galprop_keys: Sequence[str] = (
        "gas",
        "gas_psf",
        "ics",
        "ics_psf",
        "galprop_gas",
        "galprop_ics",
    ),
    iso_keys: Sequence[str] = ("iso", "isotropic"),
) -> np.ndarray:
    """
    Initialize MCMC parameters for counts-space templates.
    
    Physical templates (ps, gas, ics, iso) start at f=1 (their baseline normalization).
    Spatial shape templates (nfw, loopI, bubbles) start at 0.
    
    If iso_target_E2 is provided (legacy mode), iso will be rescaled to that target.
    """
    labels_l = [s.lower() for s in labels]
    ncomp = len(labels_l)
    f0 = np.zeros(ncomp, float)

    # Physical templates with known normalization: start at f=1
    for j, lab in enumerate(labels_l):
        if lab in tuple(k.lower() for k in ps_keys):
            f0[j] = 1.0
        if lab in tuple(k.lower() for k in galprop_keys):
            f0[j] = 1.0
        if lab in tuple(k.lower() for k in iso_keys):
            f0[j] = 1.0

    # Optional: rescale iso to target E^2 dN/dE (legacy mode)
    if iso_target_E2 is not None:
        jiso = None
        for k in iso_keys:
            k = k.lower()
            if k in labels_l:
                jiso = labels_l.index(k)
                break

        if jiso is not None:
            E2_at_f1 = infer_iso_E2dNdE_at_f1(mu[jiso], denom, Ectr_mev)
            if E2_at_f1 <= 0 or not np.isfinite(E2_at_f1):
                raise RuntimeError(f"Inferred iso E^2 dN/dE at f=1 is invalid: {E2_at_f1}")
            f0[jiso] = iso_target_E2 / E2_at_f1
    return f0


def totani_mcmc_fit(
    *,
    Cobs: np.ndarray,
    mu: np.ndarray,
    labels: Sequence[str],
    f_init: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    nwalkers: int = 64,
    nsteps: int = 5000,
    burn: int = 1000,
    thin: int = 5,
    require_autocorr: bool = False,
    early_stop: bool = False,
    autocorr_target: float = 50.0,
    autocorr_check_every: int = 1000,
    autocorr_min_steps: int = 2000,
    iso_prior_sigma_dex: Optional[float] = None,
    iso_prior_mode: str = "f",
    iso_prior_center: Optional[float] = None,
    nonstable_prior_sigma_dex: Optional[float] = None,
    nonstable_prior_centers: Optional[dict] = None,
    component_priors: Optional[dict] = None,
    init_jitter_frac: float = 1e-2,
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
    mcmc_workers: int = 1,
    verbosity: int = 2,
    poisson_negative_mode: str = "strict",
    poisson_cexp_floor: float = 1e-30,
    stat: str = "poisson",
    gaussian_flux_reg_counts: float = 1.0,
):
    def vprint(level: int, *args, **kwargs):
        if int(verbosity) >= int(level):
            print(*args, **kwargs)

    rng = rng or np.random.default_rng()

    Cobs = np.asarray(Cobs, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 2:
        raise ValueError("mu must be 2D (ncomp, npix).")
    ncomp, npix = mu.shape
    if Cobs.shape[0] != npix:
        raise ValueError(f"Cobs has {Cobs.shape[0]} entries but mu has {npix}.")

    labels = list(labels)
    if len(labels) != ncomp:
        raise ValueError("labels length must match mu.shape[0].")

    f_init = np.asarray(f_init, dtype=float)
    if f_init.shape != (ncomp,):
        raise ValueError("f_init must be shape (ncomp,).")

    if len(bounds) != ncomp:
        raise ValueError("bounds length must match ncomp.")

    labels_l = [str(x).lower() for x in labels]

    _stat = str(stat).strip().lower()
    if _stat not in ("poisson", "gaussian_flux"):
        raise ValueError(f"Unknown stat={stat!r}; choose 'poisson' or 'gaussian_flux'.")

    logprob_fn = _LogProbCallable(
        Cobs=Cobs,
        mu=mu,
        labels_l=labels_l,
        bounds=bounds,
        f_init=f_init,
        iso_prior_sigma_dex=iso_prior_sigma_dex,
        iso_prior_mode=iso_prior_mode,
        iso_prior_center=iso_prior_center,
        nonstable_prior_sigma_dex=nonstable_prior_sigma_dex,
        nonstable_prior_centers=nonstable_prior_centers,
        component_priors=component_priors,
        poisson_negative_mode=str(poisson_negative_mode),
        poisson_cexp_floor=float(poisson_cexp_floor),
        stat=_stat,
        gaussian_flux_reg_counts=float(gaussian_flux_reg_counts),
    )

    # --------------------------
    # Initialize walkers
    # --------------------------
    p0 = np.zeros((nwalkers, ncomp), dtype=float)
    scale = init_jitter_frac * (np.abs(f_init) + 1.0)

    for w in range(nwalkers):
        trial = f_init + rng.normal(0.0, scale, size=ncomp)

        for _ in range(200):
            # repair bounds component-wise
            ok = True
            for j, (lo, hi) in enumerate(bounds):
                x = trial[j]
                if (not np.isfinite(x)) or (x < lo) or (x > hi):
                    ok = False
                    x = f_init[j] + rng.normal(0.0, scale[j])
                    if np.isfinite(lo) and lo >= 0.0:
                        x = max(lo, abs(x))
                    if np.isfinite(hi):
                        x = min(hi, x)
                    trial[j] = x

            if ok and np.isfinite(logprob_fn(trial)):
                break
            trial = f_init + rng.normal(0.0, scale, size=ncomp)

        p0[w, :] = trial

    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    good = np.isfinite(lp0)
    if good.sum() < max(2, ncomp + 1):
        raise RuntimeError(f"Too few valid initial walkers: {good.sum()}/{nwalkers}")

    bad_idx = np.where(~good)[0]
    good_idx = np.where(good)[0]
    for w in bad_idx:
        src = rng.choice(good_idx)
        trial = p0[src] + rng.normal(0.0, 1e-4 * (np.abs(p0[src]) + 1.0), size=ncomp)

        for _ in range(200):
            ok = True
            for j, (lo, hi) in enumerate(bounds):
                x = trial[j]
                if (not np.isfinite(x)) or (x < lo) or (x > hi):
                    ok = False
                    x = p0[src, j] + rng.normal(0.0, 1e-4 * (abs(p0[src, j]) + 1.0))
                    if np.isfinite(lo) and lo >= 0.0:
                        x = max(lo, abs(x))
                    if np.isfinite(hi):
                        x = min(hi, x)
                    trial[j] = x
            if ok and np.isfinite(logprob_fn(trial)):
                break
            trial = p0[src] + rng.normal(0.0, 1e-4 * (np.abs(p0[src]) + 1.0), size=ncomp)

        p0[w] = trial

    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    n_bad = int(np.sum(~np.isfinite(lp0)))
    vprint(2, f"Init walkers finite: {nwalkers - n_bad}/{nwalkers}  (respawned {len(bad_idx)})")
    if n_bad:
        raise RuntimeError(f"{n_bad} walkers still have -inf logprob after respawn; check bounds/Cexp positivity.")

    # --------------------------
    # Run emcee
    # --------------------------
    used_emcee = False
    try:
        import emcee  # type: ignore
        used_emcee = True

        pool = None
        workers = int(mcmc_workers or 1)
        if workers < 1:
            raise ValueError(f"mcmc_workers must be >= 1, got {mcmc_workers!r}")
        if workers > 1:
            import multiprocessing as mp

            pool = mp.get_context("spawn").Pool(processes=workers)
            vprint(1, f"emcee worker pool: {workers} processes")

        sampler = emcee.EnsembleSampler(nwalkers, ncomp, logprob_fn, pool=pool)

        def _safe_tau_from_chain(chain_3d: np.ndarray) -> np.ndarray:
            """
            Fallback integrated autocorrelation-time estimate that is robust to
            very large parameter magnitudes by centering/scaling each walker
            chain before the FFT. Constant dimensions are treated as tau=1.
            """
            chain_3d = np.asarray(chain_3d, float)
            if chain_3d.ndim != 3:
                raise RuntimeError(f"Expected chain with ndim=3, got shape {chain_3d.shape}")

            n_step, n_walk, n_dim = chain_3d.shape
            tau = np.ones(n_dim, dtype=float)

            def _acf_1d(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, float)
                x = x[np.isfinite(x)]
                n = x.size
                if n < 2:
                    return np.ones(1, dtype=float)
                x = x - np.mean(x)
                s = float(np.std(x))
                if (not np.isfinite(s)) or s <= 0.0:
                    return np.ones(n, dtype=float)
                x = x / s
                n_fft = 1
                while n_fft < 2 * n:
                    n_fft *= 2
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    f = np.fft.fft(x, n=n_fft)
                    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
                if (acf.size == 0) or (not np.isfinite(acf[0])) or (acf[0] <= 0.0):
                    return np.ones(n, dtype=float)
                acf = acf / acf[0]
                acf[~np.isfinite(acf)] = 0.0
                return acf

            for j in range(n_dim):
                xjw = chain_3d[:, :, j]
                if not np.any(np.isfinite(xjw)):
                    tau[j] = np.inf
                    continue
                if float(np.nanstd(xjw)) <= 0.0:
                    tau[j] = 1.0
                    continue

                acf_sum = None
                n_used = 0
                for w in range(n_walk):
                    acf_w = _acf_1d(xjw[:, w])
                    if acf_sum is None:
                        acf_sum = np.zeros_like(acf_w, dtype=float)
                    if acf_w.shape[0] != acf_sum.shape[0]:
                        m = min(acf_sum.shape[0], acf_w.shape[0])
                        acf_sum = acf_sum[:m]
                        acf_w = acf_w[:m]
                    acf_sum += acf_w
                    n_used += 1

                if (acf_sum is None) or (n_used == 0):
                    tau[j] = np.inf
                    continue

                acf = acf_sum / float(n_used)
                tau_j = 1.0
                for t in range(1, acf.shape[0]):
                    if (not np.isfinite(acf[t])) or (acf[t] <= 0.0):
                        break
                    tau_j += 2.0 * float(acf[t])
                    if t >= 5.0 * tau_j:
                        break
                tau[j] = max(float(tau_j), 1.0)

            return tau

        def _estimate_tau(*, sampler, verbosity: int) -> np.ndarray:
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    tau = np.asarray(sampler.get_autocorr_time(tol=0), float)
                if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0):
                    raise RuntimeError(f"Non-finite tau from emcee autocorr estimate: {tau}")
                return tau
            except Exception as e:
                if int(verbosity) >= 2:
                    vprint(2, f"falling back to safe autocorr estimate: {repr(e)}")
                tau = _safe_tau_from_chain(sampler.get_chain())
                if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0):
                    raise RuntimeError(f"Fallback autocorr estimate failed: {tau}") from e
                return tau

        def _report_autocorr_status(*, n_done: int):
            try:
                tau = _estimate_tau(sampler=sampler, verbosity=int(verbosity))
                n_over_tau = float(n_done) / tau
                if np.any(~np.isfinite(n_over_tau)):
                    raise RuntimeError(f"Non-finite N/tau encountered: {n_over_tau}")
                ok = bool(np.all(n_over_tau >= float(autocorr_target)))
                if int(verbosity) >= 2:
                    vprint(2, "tau:", tau)
                    vprint(2, "N/tau:", n_over_tau)
                if int(verbosity) >= 1:
                    min_not = float(np.min(n_over_tau))
                    vprint(1, f"autocorr status: {'achieved' if ok else 'NOT achieved'} (min N/tau={min_not:.3f})")
            except Exception as e:
                if int(verbosity) >= 2:
                    vprint(2, "autocorr time not available yet:", repr(e))
                if int(verbosity) >= 1:
                    vprint(1, "autocorr status: unavailable")

        try:
            if early_stop:
                total = int(nsteps)
                step = max(1, int(autocorr_check_every))
                min_steps = int(autocorr_min_steps)
                state = None
                n_done = 0

                while n_done < total:
                    n_chunk = min(step, total - n_done)
                    run_kwargs = {}
                    if state is not None:
                        # emcee can spuriously reject valid restart states in chunked runs
                        # when walkers have contracted; we already validated the first state.
                        run_kwargs["skip_initial_state_check"] = True
                    state = sampler.run_mcmc(
                        p0 if state is None else state,
                        int(n_chunk),
                        progress=progress,
                        **run_kwargs,
                    )
                    n_done += int(n_chunk)

                    if n_done < min_steps or n_done <= burn:
                        continue

                    try:
                        tau = _estimate_tau(sampler=sampler, verbosity=int(verbosity))
                        n_over_tau = n_done / tau
                        if np.any(~np.isfinite(n_over_tau)):
                            raise RuntimeError(f"Non-finite N/tau encountered: {n_over_tau}")
                        if np.all(n_over_tau >= float(autocorr_target)):
                            vprint(2, "tau:", tau)
                            vprint(2, "N/tau:", n_over_tau)
                            vprint(2, f"Early stop: reached N/tau >= {float(autocorr_target):g} at N={n_done}")
                            break
                    except Exception as e:
                        if bool(require_autocorr):
                            raise RuntimeError(f"Autocorr convergence check requested but failed: {repr(e)}") from e

                _report_autocorr_status(n_done=int(n_done))
            else:
                sampler.run_mcmc(p0, int(nsteps), progress=progress)
                _report_autocorr_status(n_done=int(nsteps))
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain = sampler.get_chain()
        logprob = sampler.get_log_prob()
        acc = sampler.acceptance_fraction

    except Exception:
        raise  # keep your fallback if you want; but this is usually better to fail loudly here

    # --------------------------
    # Summarize posterior
    # --------------------------
    nsteps_eff = int(chain.shape[0])
    if burn >= nsteps_eff:
        raise ValueError(f"burn must be < nsteps_done. burn={burn}, nsteps_done={nsteps_eff}")

    post = chain[burn::thin, :, :].reshape(-1, ncomp)
    post_lp = logprob[burn::thin, :].reshape(-1)
    if post.shape[0] == 0:
        raise RuntimeError(f"No posterior samples after burn/thin. nsteps_done={nsteps_eff}, burn={burn}, thin={thin}.")

    imax = int(np.nanargmax(post_lp))
    f_ml = post[imax].copy()
    loglike_ml = float(post_lp[imax])

    q16, q50, q84 = np.quantile(post, [0.16, 0.50, 0.84], axis=0)

    return MCMCResult(
        labels=labels,
        f_ml=f_ml,
        f_p16=q16,
        f_p50=q50,
        f_p84=q84,
        loglike_ml=loglike_ml,
        chain=chain,
        logprob=logprob,
        acceptance_fraction=acc,
        used_emcee=used_emcee,
    )
