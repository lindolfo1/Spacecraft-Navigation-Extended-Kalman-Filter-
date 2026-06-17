import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────────────
# G * M_sun expressed in AU³ / s²
#   G  = 6.674e-11  m³ kg⁻¹ s⁻²
#   M☉ = 1.989e30   kg
#   1 AU = 1.496e11  m
#   GM_sun [m³/s²] = 6.674e-11 * 1.989e30 = 1.327e20
#   Convert to AU³/s²: 1.327e20 / (1.496e11)³ ≈ 3.964e-14  AU³/s²
GM = 3.964e-14   # AU³ s⁻²   (G × M_sun, M already baked in)

#  Default is 86400 (1 day)
step_size=86_400

# ── Orbital-velocity helper ───────────────────────────────────────────────────
def circular_velocity(r_au: float) -> float:
    """Return the circular-orbit speed (AU/s) at radius r_au (AU)."""
    return np.sqrt(GM / r_au)


# ── Derivative (equations of motion) ─────────────────────────────────────────
# state = [x, y, z, vx, vy, vz]   positions in AU, velocities in AU/s
def calculate_derivative(state: np.ndarray) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1e-6:
        raise ValueError("Collision with the Sun detected (r < 1e-6 AU)")
    factor = -GM / r**3
    ax, ay, az = factor*x, factor*y, factor*z
    return np.array([vx, vy, vz, ax, ay, az])


# ── RK4 integrator ────────────────────────────────────────────────────────────
def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = calculate_derivative(state)
    k2 = calculate_derivative(state + (dt / 2) * k1)
    k3 = calculate_derivative(state + (dt / 2) * k2)
    k4 = calculate_derivative(state + dt * k3)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ── Orbit integrator ──────────────────────────────────────────────────────────
def compute_orbit(
    state: np.ndarray,
    num_steps: int = 1000,
    verbose: bool = False,
) -> list[np.ndarray]:
    path = [state]
    if verbose:
        print("step   x (AU)    y (AU)    z (AU)  |  vx       vy       vz")
    for step in range(1, num_steps):
        state = rk4_step(path[-1], step_size)
        path.append(state)
        if verbose and step % 100 == 0:
            print(
                f"{step:4d}  ({state[0]:8.4f}, {state[1]:8.4f}, {state[2]:8.4f})"
                f"  ({state[3]:.4e}, {state[4]:.4e}, {state[5]:.4e})"
            )
    return path


# ── Plotter ───────────────────────────────────────────────────────────────────
def plot_orbit_three_views(path: list[np.ndarray], filename: str = "orbit_views.png"):
    """Save three 2-D projections of the orbit (XY, XZ, YZ planes)."""
    arr = np.array(path)
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    specs = [
        (axes[0], x, y, "X (AU)", "Y (AU)", "XY Plane (Top View)",   "blue"),
        (axes[1], x, z, "X (AU)", "Z (AU)", "XZ Plane (Side View)",  "green"),
        (axes[2], y, z, "Y (AU)", "Z (AU)", "YZ Plane (Front View)", "purple"),
    ]
    for ax, a, b, xl, yl, title, colour in specs:
        ax.plot(a, b, color=colour, linewidth=1.5)
        ax.plot(0, 0, "ro", markersize=10, label="Sun")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.set_aspect("equal"); ax.grid(True); ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Orbit state generator ─────────────────────────────────────────────────────
def make_orbit_state(
    a: float,
    e: float = 0.0,
    inclination: float = 0.0,
    lon_ascending_node: float = 0.0,
    arg_periapsis: float = 0.0,
) -> np.ndarray:
    """
    Build an initial [x, y, z, vx, vy, vz] state at periapsis from
    standard Keplerian orbital elements.

    Parameters
    ----------
    a : float
        Semi-major axis (AU).  Controls the overall size of the orbit.

    e : float  [0, 1)
        Eccentricity.
          0   → perfect circle
          0.5 → moderately elliptical  (e.g. typical comet)
          →1  → increasingly elongated (must stay < 1 for a bound orbit)

    inclination : float  (degrees)
        Tilt of the orbital plane relative to the XY reference plane.
          0°  → orbit lies flat in the XY plane
          90° → orbit is edge-on (tilted up into the XZ plane)

    lon_ascending_node : float  (degrees)  — "Ω"
        Rotates the line where the orbit crosses the XY plane.
        Spins the whole tilted orbit around the Z axis.

    arg_periapsis : float  (degrees)  — "ω"
        Rotates the ellipse *within* its own plane, moving the closest
        approach point (periapsis) around the orbit.

    Returns
    -------
    np.ndarray  shape (6,)
        [x, y, z, vx, vy, vz] in AU and AU/s, starting at periapsis.

    Notes
    -----
    The three angles correspond to the standard Euler rotation sequence
    used in celestial mechanics (Ω → i → ω).  The Sun sits at the origin.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError(f"Eccentricity must be in [0, 1); got {e}")
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive; got {a}")

    # ── 1. Periapsis distance and speeds in the orbital plane ─────────────────
    r_peri = a * (1.0 - e)          # closest approach distance (AU)

    # Vis-viva: v² = GM (2/r − 1/a)
    v_peri = np.sqrt(GM * (2.0 / r_peri - 1.0 / a))   # AU/s

    # In the unrotated orbital plane: body starts at periapsis on +x′ axis,
    # moving in the +y′ direction.
    pos_orb = np.array([r_peri, 0.0, 0.0])
    vel_orb = np.array([0.0,  v_peri, 0.0])

    # ── 2. Three Euler rotations: ω → i → Ω ──────────────────────────────────
    i   = np.radians(inclination)
    Omega = np.radians(lon_ascending_node)
    omega = np.radians(arg_periapsis)

    def Rz(angle):
        """Rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[ c, -s, 0],
                         [ s,  c, 0],
                         [ 0,  0, 1]])

    def Rx(angle):
        """Rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]])

    # Standard celestial-mechanics sequence
    R = Rz(Omega) @ Rx(i) @ Rz(omega)

    pos = R @ pos_orb
    vel = R @ vel_orb

    return np.concatenate([pos, vel])


def period_days(a: float) -> float:
    """Keplerian orbital period in days for semi-major axis a (AU)."""
    # T² = (4π²/GM) a³   →   T in seconds, convert to days
    T_s = 2 * np.pi * np.sqrt(a**3 / GM)
    return T_s / 86_400


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Example 1: circular, flat orbit at 3.9 AU ────────────────────────────
    s1 = make_orbit_state(a=3.9, e=0.0)
    print(f"Example 1 — circular 3.9 AU | period = {period_days(3.9):.1f} days")

    # ── Example 2: eccentric + tilted (comet-like) ───────────────────────────
    s2 = make_orbit_state(
        a=3.9,
        e=0.6,              # quite elongated
        inclination=30.0,   # 30° tilt out of the ecliptic
        lon_ascending_node=45.0,
        arg_periapsis=20.0,
    )
    print(f"Example 2 — e=0.6, i=30°   | period = {period_days(3.9):.1f} days")

    # Integrate each for 2 full periods
    steps = int(2 * period_days(3.9))   # one step per day
    o1 = compute_orbit(s1, num_steps=steps)
    o2 = compute_orbit(s2, num_steps=steps)

    # ── Combined plot ─────────────────────────────────────────────────────────
    a1 = np.array(o1);  a2 = np.array(o2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    views = [
        (0, 1, "X (AU)", "Y (AU)", "XY Plane (Top View)"),
        (0, 2, "X (AU)", "Z (AU)", "XZ Plane (Side View)"),
        (1, 2, "Y (AU)", "Z (AU)", "YZ Plane (Front View)"),
    ]
    for ax, (ci, cj, xl, yl, title) in zip(axes, views):
        ax.plot(a1[:, ci], a1[:, cj], color="steelblue",  lw=1.5, label="circular, flat")
        ax.plot(a2[:, ci], a2[:, cj], color="darkorange", lw=1.5, label="e=0.6, i=30°")
        ax.plot(0, 0, "ro", markersize=10, label="Sun")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)

    plt.tight_layout()
    out = "orbit_views.png"
    plt.savefig(out, dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)