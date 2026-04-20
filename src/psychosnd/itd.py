import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre


def _hankel2_deriv(n, x):
    return spherical_jn(n, x, derivative=True) - 1j * spherical_yn(n, x, derivative=True)


def _pressure(theta_prime, ka, N):
    """Complex pressure at angle theta_prime on a rigid sphere (Aaronson & Hartmann 2014, Eq. 2).

    The (p_o/ka)^2 prefactor is omitted as it cancels when computing the ITD ratio.
    """
    cos_t = np.cos(theta_prime)
    return sum(
        (1j ** (n + 1)) * (2 * n + 1) * eval_legendre(n, cos_t) / _hankel2_deriv(n, ka)
        for n in range(N + 1)
    )


def _n_terms(ka, N_terms):
    if N_terms is not None:
        return N_terms
    return int(np.ceil(2 * ka + 10))


def itd_woodworth(angles_deg, head_radius=0.0875, c=343.0):
    """Woodworth model ITD (frequency-independent, high-frequency limit).

    Parameters
    ----------
    angles_deg : array-like
        Azimuth angles in degrees (0 = front, 90 = right).
    head_radius : float
        Head radius in metres (default 0.0875 m).
    c : float
        Speed of sound in m/s (default 343.0).

    Returns
    -------
    itd : ndarray
        ITD in seconds. Positive means source is to the right.
    """
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    itd = np.where(
        np.abs(theta) <= np.pi / 2,
        (head_radius / c) * (theta + np.sin(theta)),
        (head_radius / c) * (np.sign(theta) * np.pi - theta + np.sin(theta)),
    )
    return itd


def itd_diffraction(frequencies, angles_deg, head_radius=0.0875, c=343.0, theta_E=np.pi / 2, N_terms=None):
    """ITD via exact spherical head diffraction (Aaronson & Hartmann 2014, Eqs. 2 & 3).

    Implements ITD = Im{ln[p(θ_E+θ)/p(θ_E-θ)]} / ω (Eq. 3) where p is the partial-wave
    pressure sum (Eq. 2). The interaural phase difference (IPD) exceeds π above ~1 kHz for
    large angles, so the phase is tracked continuously from a low-frequency reference via
    np.unwrap before dividing by ω to recover the true phase delay.

    The internal frequency grid uses ≤200 Hz spacing, which limits consecutive phase steps
    to < π (max ITD ≈ 800 µs → max step ≈ 2π × 800e-6 × 200 ≈ 1.0 rad).

    Parameters
    ----------
    frequencies : array-like
        Frequencies in Hz.
    angles_deg : array-like
        Azimuth angles in degrees (0 = front, 90 = right).
    head_radius : float
        Head radius in metres (default 0.0875 m = 87.5 mm).
    c : float
        Speed of sound in m/s (default 343.0).
    theta_E : float
        Ear angle in radians (default π/2 for antipodal ears).
    N_terms : int or None
        Partial-wave sum truncation. Auto-selected from ka when None
        using N = ceil(2*ka + 10); at 20 kHz this yields ~74 terms.

    Returns
    -------
    itd : ndarray, shape (len(frequencies), len(angles_deg))
        ITD in seconds. Positive values mean source is to the right.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))

    f_max = float(np.max(frequencies))
    # 200 Hz spacing limits phase steps to ~1 rad < π for ITD ≤ 800 µs.
    n_grid = max(int(np.ceil(f_max / 200)) + 2, 20)
    # Start well above DC to avoid near-singular Hankel denominators.
    freq_grid = np.linspace(f_max / n_grid, f_max, n_grid)

    itd = np.zeros((len(frequencies), len(angles_rad)))

    for j, theta in enumerate(angles_rad):
        phases = np.empty(len(freq_grid))
        for k, fg in enumerate(freq_grid):
            ka = 2 * np.pi * fg * head_radius / c
            N = _n_terms(ka, N_terms)
            p_far = _pressure(theta_E + theta, ka, N)
            p_near = _pressure(theta_E - theta, ka, N)
            phases[k] = np.angle(p_far / p_near)

        # Unwrap across frequency so the phase is continuous.
        unwrapped = np.unwrap(phases)

        for i, f in enumerate(frequencies):
            phi = np.interp(f, freq_grid, unwrapped)
            # Negative sign: h_n^(2) uses the e^{-iωt} convention, giving
            # arg(p_L/p_R) = −ω·ITD, so ITD = −Φ/ω.
            itd[i, j] = -phi / (2 * np.pi * f)

    return itd
