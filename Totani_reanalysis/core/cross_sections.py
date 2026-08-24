import numpy as np

# --- constants ---
alpha = 1/137.035999084
v = 246.0
mW, mH, GammaH = 80.379, 125.25, 4.07e-3
mtop = 172.5
Nc_top, Q_top = 3, 2/3

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb
FB_TO_CM2      = 1e-39         # 1 fb = 1e-39 cm^2

def get_flat_grav_cross_sections(omega_GeV, m_chi_GeV, theta_rad, lam):
    # Flat-space cross sections (fb/sr)
    # omega_GeV can be a scalar or array
    K     = np.asarray(omega_GeV)  # dimensionless photon energy  (= omega * r_s = 1.0)
    P     = m_chi_GeV              # DM mass in GeV (r_s=1 so GeV is the energy unit)
    lam4  = lam**4                 # GeV^-4  (lam = sqrt(8 pi G_N), G_N in GeV^-2)
    cos_t = np.cos(theta_rad)
    
    # Flat-space prefactor: lambda^4 (K+P)^4 / [16 (2pi)^2 P^2]
    # Dimension: GeV^-4 * GeV^4 / GeV^2 = GeV^-2
    flat_pre = lam4 * (K + P)**4 / (16.0 * (2*np.pi)**2 * P**2)
    
    # Angular factors (eqs 5.18-5.19 in thesis)
    denom = (1.0 - cos_t)**2

    scale = GEV2_TO_FB
    
    dsig_flat_perp = flat_pre / denom                  * scale   # fb/sr
    dsig_flat_par  = flat_pre * cos_t**2 / denom       * scale   # fb/sr
    dsig_flat_unpol = 0.5 * (dsig_flat_perp + dsig_flat_par)      # fb/sr
    
    return dsig_flat_perp, dsig_flat_par, dsig_flat_unpol

# ---------- Loop functions (H -> γγ) ----------

def get_weak_y_eff(mchi):
    gW = 2.0 * mW / v
    return np.sqrt((alpha**2) * (gW**4) * (mchi**2) / ((4.0*np.pi)**2 * (mW**4)))

def f_scalar(tau):
    """
    f(τ) loop function.
    Works for scalar or numpy array τ (real or complex).
    """
    tau_arr = np.asarray(tau, dtype=complex)
    out = np.zeros_like(tau_arr, dtype=complex)

    mask_real = tau_arr.real >= 1.0
    if np.any(mask_real):
        out[mask_real] = np.arcsin(1.0 / np.sqrt(tau_arr[mask_real]))**2

    mask_cmplx = ~mask_real
    if np.any(mask_cmplx):
        root = np.sqrt(1.0 - tau_arr[mask_cmplx])
        out[mask_cmplx] = -0.25 * (
            np.log((1 + root) / (1 - root)) - 1j*np.pi
        )**2

    if np.ndim(tau) == 0:
        return out.item()
    return out

def A1_over2(tau):  # spin-1/2
    tau_arr = np.asarray(tau, dtype=complex)
    return -2.0 * tau_arr * (1.0 + (1.0 - tau_arr) * f_scalar(tau_arr))

def A1(tau):        # spin-1
    tau_arr = np.asarray(tau, dtype=complex)
    return 2.0 + 3.0 * tau_arr + 3.0 * (2.0 * tau_arr - tau_arr**2) * f_scalar(tau_arr)

def IW_IF_from_t(t, mW_, mferm_, eps=1e-18):
    """
    Given Mandelstam t, return (I_W, I_F) loop integrals.
    """
    t = np.asarray(t, dtype=float)
    IW = np.zeros_like(t, dtype=complex)
    IF = np.zeros_like(t, dtype=complex)
    safe = ~np.isclose(t, 0.0, atol=eps, rtol=0.0)
    if np.any(safe):
        betaW = -4.0*(mW_**2)/t[safe]
        betaf = -4.0*(mferm_**2)/t[safe]
        IW[safe] = A1(betaW)
        IF[safe] = A1_over2(betaf)
    return (IW.item(), IF.item()) if IW.ndim == 0 else (IW, IF)


# ---------- Kinematics ----------

def get_s_cm(mchi, k):
    """CoM frame: s = (sqrt(mchi^2 + k^2) + k)^2, with k = E_gamma."""
    return (np.sqrt(mchi**2 + k**2) + k)**2

def get_s_lab_DMrest(mchi, omega):
    """Lab frame (DM at rest): s = mchi^2 + 2 mchi * omega."""
    return mchi**2 + 2*mchi*omega

def get_t_cm(Eg, theta):
    """CoM frame: t = -2 Eg^2 (1 - cos(theta))."""
    return -2.0 * Eg**2 * (1.0 - np.cos(theta))

def get_t_lab_DMrest(mchi, omega, theta):
    """
    Lab frame (DM at rest): Compton-like kinematics with m_e -> mchi.
    omega' = omega / (1 + (omega/mchi)*(1-cosθ))
    t = -2 * omega * omega' * (1 - cosθ)
    """
    denom = 1.0 + (omega/mchi)*(1.0 - np.cos(theta))
    omega_out = omega / denom
    return -2.0 * omega * omega_out * (1.0 - np.cos(theta))

def lab_recoil_ratio(mchi, omega, theta):
    """Photon energy ratio omega'/omega in the DM-rest lab frame."""
    denom = 1.0 + (omega/mchi)*(1.0 - np.cos(theta))
    return 1.0 / denom


# ---------- dσ/dΩ ----------

def get_flat_weak_cross_sections(
    mchi,
    y_eff, # effective Higgs-portal coupling
    theta,
    E_gamma,
    *,         
    frame="cm",            # "cm" or "lab"
    in_SI=False,
    mW_=mW, mH_=mH, GammaH_=GammaH,
    mferm_=mtop, Nc_=Nc_top, Qf_=Q_top,
    which="full"
):
    """
    mchi      : DM mass [GeV]
    theta     : scattering angle [rad]
    E_gamma   : photon energy [GeV]
    y_eff  : effective Higgs-portal coupling (free parameter)

    returns   : fb/sr (default) or m^2/sr if in_SI=True
    """
    # --- kinematics ---
    if frame == "lab":
        s = get_s_lab_DMrest(mchi, E_gamma)
        t = get_t_lab_DMrest(mchi, E_gamma, theta)
    else:
        s = get_s_cm(mchi, E_gamma)
        t = get_t_cm(E_gamma, theta)

    # --- loops ---
    gW = 2.0 * mW_ / v
    IW, IF = IW_IF_from_t(t, mW_, mferm_)

    if which == "F":   # fermion only (top)
        amp = Nc_ * (Qf_**2) * IF
    elif which == "W": # W only
        amp = IW
    else:              # both
        amp = IW + Nc_ * (Qf_**2) * IF

    amp2 = (amp * np.conjugate(amp)).real

    # Prefactor ∝ ychi_eff^2
    pref = (alpha**2 * gW**4 * y_eff**2) / ((4.0 * np.pi)**2 * mW_**4)

    tpart = 3.0 * (t**2) / 8.0
    prop  = (2.0*mchi**2 - 0.5*t) / (((t - mH_**2)**2) + (mH_**2)*(GammaH_**2))
    if frame == "lab":
        phase = amp2 * lab_recoil_ratio(mchi, E_gamma, theta)**2 / (
            64.0 * (np.pi**2) * mchi**2
        )
    else:
        phase = amp2 / (64.0 * (np.pi**2) * s)

    val = pref * tpart * prop * phase   # GeV^-2 / sr

    return val * (HC2_GEV2_TO_M2 if in_SI else GEV2_TO_FB)

# σ(E) from dσ/dΩ
def sigma_tot_weak(E_gamma, mchi, y_eff, n_theta=300):
    """
    Total cross section σ(E) = ∫ dΩ (dσ/dΩ)
    E_gamma : scalar (GeV)
    mchi    : scalar (GeV)
    returns σ (m^2)
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]

    dsdo = get_flat_weak_cross_sections_vec(
        mchi, y_eff, theta, E_gamma,
        frame="lab", in_SI=True, which="full"
    )  # ychi_eff=1 for the shape
    dsdo = np.nan_to_num(dsdo, nan=0.0, posinf=0.0, neginf=0.0)

    integral = 2.0 * np.pi * np.sum(np.sin(theta) * dsdo) * dtheta
    return integral  # m^2

sigma_tot_weak_vec = np.vectorize(sigma_tot_weak)
get_flat_weak_cross_sections_vec = np.vectorize(
    lambda mchi, y, theta, E, **kw: get_flat_weak_cross_sections(mchi, y, theta, E, **kw)
)
