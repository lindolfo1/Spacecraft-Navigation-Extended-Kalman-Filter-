import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(ax, mean, cov2d, n_std=2.0, **kwargs):
    """Draw a 2D confidence ellipse from a 2x2 covariance slice."""
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_ekf(states, P_list, orbit_path, n_ellipses=20, output_path="ekf_plot.png"):
    """
    Plot EKF results: 3D trajectory, XY/XZ projections with uncertainty ellipses,
    and position error over time.

    Parameters:
        states      : list of np.ndarray shape (6,)  — EKF estimated states
        P_list      : list of np.ndarray shape (6,6) — covariance at each step
        orbit_path  : list/array shape (N,6)         — true orbit states
        n_ellipses  : int  — how many uncertainty ellipses to draw
        output_path : str  — output PNG path
    """
    states     = np.array(states)
    orbit_path = np.array(orbit_path)
    n          = min(len(states), len(orbit_path))
    states     = states[:n]
    orbit_path = orbit_path[:n]
    P_list     = P_list[:n]
    t          = np.arange(n)

    est_pos  = states[:, :3]
    true_pos = orbit_path[:, :3]
    err      = est_pos - true_pos
    err_norm = np.linalg.norm(err, axis=1)

    # ellipse indices spread evenly across the trajectory
    ellipse_idx = np.linspace(0, n - 1, n_ellipses, dtype=int)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("EKF Spacecraft Navigation", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. 3D trajectory ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :], projection="3d")
    ax1.plot(*true_pos.T, color="steelblue", linewidth=1.5, label="True orbit",    alpha=0.8)
    ax1.plot(*est_pos.T,  color="tomato",    linewidth=1.2, label="EKF estimate",  alpha=0.8, linestyle="--")
    ax1.scatter(*true_pos[0],  color="green",  s=40, zorder=5, label="Start")
    ax1.scatter(*true_pos[-1], color="purple", s=40, zorder=5, label="End")
    ax1.set_xlabel("X (AU)"); ax1.set_ylabel("Y (AU)"); ax1.set_zlabel("Z (AU)")
    ax1.set_title("3D trajectory", fontsize=10)
    ax1.legend(fontsize=8, ncol=4)

    # ── 2. XY projection + uncertainty ellipses ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(true_pos[:, 0], true_pos[:, 1], color="steelblue", linewidth=1.5, label="True", alpha=0.8)
    ax2.plot(est_pos[:, 0],  est_pos[:, 1],  color="tomato",    linewidth=1.0, label="EKF",  alpha=0.8, linestyle="--")
    for i in ellipse_idx:
        cov2d = P_list[i][np.ix_([0, 1], [0, 1])]
        confidence_ellipse(ax2, (est_pos[i, 0], est_pos[i, 1]), cov2d,
                           n_std=2, edgecolor="tomato", facecolor="none",
                           linewidth=0.6, alpha=0.5)
    ax2.scatter(0, 0, color="gold", s=80, zorder=5, marker="*", label="Sun")
    ax2.set_xlabel("X (AU)"); ax2.set_ylabel("Y (AU)")
    ax2.set_title("XY plane  (2σ ellipses)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_aspect("equal"); ax2.spines[["top", "right"]].set_visible(False)

    # ── 3. XZ projection + uncertainty ellipses ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(true_pos[:, 0], true_pos[:, 2], color="steelblue", linewidth=1.5, label="True", alpha=0.8)
    ax3.plot(est_pos[:, 0],  est_pos[:, 2],  color="tomato",    linewidth=1.0, label="EKF",  alpha=0.8, linestyle="--")
    for i in ellipse_idx:
        cov2d = P_list[i][np.ix_([0, 2], [0, 2])]
        confidence_ellipse(ax3, (est_pos[i, 0], est_pos[i, 2]), cov2d,
                           n_std=2, edgecolor="tomato", facecolor="none",
                           linewidth=0.6, alpha=0.5)
    ax3.scatter(0, 0, color="gold", s=80, zorder=5, marker="*", label="Sun")
    ax3.set_xlabel("X (AU)"); ax3.set_ylabel("Z (AU)")
    ax3.set_title("XZ plane  (2σ ellipses)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.set_aspect("equal"); ax3.spines[["top", "right"]].set_visible(False)

    # ── 4. Position error magnitude ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, err_norm, color="#8E24AA", linewidth=1.4)
    ax4.set_title("|Position error|  ‖est − true‖", fontsize=10)
    ax4.set_xlabel("Time step"); ax4.set_ylabel("Error (AU)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)

    # ── 5. Per-axis error ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, err[:, 0], label="X err", linewidth=1.2, color="steelblue")
    ax5.plot(t, err[:, 1], label="Y err", linewidth=1.2, color="tomato")
    ax5.plot(t, err[:, 2], label="Z err", linewidth=1.2, color="seagreen")
    ax5.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax5.set_title("Per-axis position error", fontsize=10)
    ax5.set_xlabel("Time step"); ax5.set_ylabel("Error (AU)")
    ax5.legend(fontsize=8); ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.spines[["top", "right"]].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_comparison(states, naive_pos, orbit_path, output_path="comparison_plot.png"):
    """
    Compare naive positioning vs EKF estimate vs true orbit.

    Parameters:
        states      : list of np.ndarray shape (6,)  — EKF estimated states
        naive_pos   : list of [x, y, z]              — naive position estimates
        orbit_path  : list/array shape (N,6)         — true orbit states
        output_path : str                            — output PNG path
    """
    states     = np.array(states)
    naive_pos  = np.array(naive_pos)
    orbit_path = np.array(orbit_path)

    n         = min(len(states), len(naive_pos), len(orbit_path))
    est_pos   = states[:n, :3]
    naive_pos = naive_pos[:n]
    true_pos  = orbit_path[:n, :3]
    t         = np.arange(n)

    ekf_err   = np.linalg.norm(est_pos  - true_pos, axis=1)
    naive_err = np.linalg.norm(naive_pos - true_pos, axis=1)

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle("Naive Positioning vs EKF", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. XY trajectories ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_pos[:, 0],  true_pos[:, 1],  color="steelblue", linewidth=1.8, label="True",  alpha=0.9)
    ax1.plot(naive_pos[:, 0], naive_pos[:, 1], color="darkorange", linewidth=1.0, label="Naive", alpha=0.7, linestyle=":")
    ax1.plot(est_pos[:, 0],   est_pos[:, 1],   color="tomato",    linewidth=1.0, label="EKF",   alpha=0.8, linestyle="--")
    ax1.scatter(0, 0, color="gold", s=80, zorder=5, marker="*")
    ax1.set_xlabel("X (AU)"); ax1.set_ylabel("Y (AU)")
    ax1.set_title("XY plane", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_aspect("equal"); ax1.spines[["top", "right"]].set_visible(False)

    # ── 2. XZ trajectories ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(true_pos[:, 0],  true_pos[:, 2],  color="steelblue", linewidth=1.8, label="True",  alpha=0.9)
    ax2.plot(naive_pos[:, 0], naive_pos[:, 2], color="darkorange", linewidth=1.0, label="Naive", alpha=0.7, linestyle=":")
    ax2.plot(est_pos[:, 0],   est_pos[:, 2],   color="tomato",    linewidth=1.0, label="EKF",   alpha=0.8, linestyle="--")
    ax2.scatter(0, 0, color="gold", s=80, zorder=5, marker="*")
    ax2.set_xlabel("X (AU)"); ax2.set_ylabel("Z (AU)")
    ax2.set_title("XZ plane", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_aspect("equal"); ax2.spines[["top", "right"]].set_visible(False)

    # ── 3. Error magnitude comparison ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t, naive_err, color="darkorange", linewidth=1.4, label="Naive |error|", alpha=0.8)
    ax3.plot(t, ekf_err,   color="tomato",    linewidth=1.4, label="EKF |error|",   alpha=0.9)
    ax3.set_title("|Position error|  ‖est − true‖", fontsize=10)
    ax3.set_xlabel("Time step"); ax3.set_ylabel("Error (AU)")
    ax3.legend(fontsize=8); ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── 4. Per-axis: Naive ────────────────────────────────────────────────
    naive_xyz_err = naive_pos - true_pos
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, naive_xyz_err[:, 0], color="steelblue", linewidth=1.2, label="X")
    ax4.plot(t, naive_xyz_err[:, 1], color="tomato",    linewidth=1.2, label="Y")
    ax4.plot(t, naive_xyz_err[:, 2], color="seagreen",  linewidth=1.2, label="Z")
    ax4.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax4.set_title("Naive per-axis error", fontsize=10)
    ax4.set_xlabel("Time step"); ax4.set_ylabel("Error (AU)")
    ax4.legend(fontsize=8); ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)

    # ── 5. Per-axis: EKF ──────────────────────────────────────────────────
    ekf_xyz_err = est_pos - true_pos
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, ekf_xyz_err[:, 0], color="steelblue", linewidth=1.2, label="X")
    ax5.plot(t, ekf_xyz_err[:, 1], color="tomato",    linewidth=1.2, label="Y")
    ax5.plot(t, ekf_xyz_err[:, 2], color="seagreen",  linewidth=1.2, label="Z")
    ax5.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax5.set_title("EKF per-axis error", fontsize=10)
    ax5.set_xlabel("Time step"); ax5.set_ylabel("Error (AU)")
    ax5.legend(fontsize=8); ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.spines[["top", "right"]].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")