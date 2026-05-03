#!/usr/bin/env python3
""" Project 3 code: constrained cart-pole trajopt + TVLQR.

Run:
    python3 project3_simple.py
    python3 project3_simple.py --no-show

Outputs:
    outputs/figures/trajectory_and_tracking.png
    outputs/figures/lqr_R_ablation.png
    outputs/optimized_trajectory.npz
    outputs/metrics.json
    A real-time Matplotlib animation window, unless --no-show is used.

State:
    z = [x, theta, x_dot, theta_dot]
    theta = 0  : upright equilibrium
    theta = pi : hanging initial pose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_discrete_are
from scipy.optimize import Bounds, NonlinearConstraint, minimize


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
STUDENT_NAME = "李皓宁"
STUDENT_ID = "2023112386"


@dataclass(frozen=True)
class CartPoleParams:
    mc: float = 1.0
    mp: float = 0.2
    l: float = 0.5
    g: float = 9.81
    x_limit: float = 2.4
    u_limit: float = 30.0


@dataclass(frozen=True)
class TrajOptConfig:
    horizon: float = 4.0
    knots: int = 51
    q_x: float = 0.05
    q_theta: float = 0.1
    q_v: float = 0.01
    r_u: float = 0.003


P = CartPoleParams()
CFG = TrajOptConfig()
X0 = np.array([0.0, math.pi, 0.0, 0.0])
XF = np.array([0.0, 0.0, 0.0, 0.0])


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def continuous_angle(theta, reference_start=None):
    theta = np.unwrap(theta)
    if reference_start is not None:
        theta = theta + 2.0 * np.pi * round((reference_start - theta[0]) / (2.0 * np.pi))
    return theta


def dynamics(z, u, p=P):
    """Continuous cart-pole dynamics from Lagrange equations."""
    _, th, xd, thd = z
    c, s = math.cos(th), math.sin(th)
    mass = np.array(
        [
            [p.mc + p.mp, p.mp * p.l * c],
            [p.mp * p.l * c, p.mp * p.l**2],
        ]
    )
    rhs = np.array(
        [
            u + p.mp * p.l * s * thd**2,
            p.mp * p.g * p.l * s,
        ]
    )
    xdd, thdd = np.linalg.solve(mass, rhs)
    return np.array([xd, thd, xdd, thdd])


def rk4_step(z, u, dt):
    k1 = dynamics(z, u)
    k2 = dynamics(z + 0.5 * dt * k1, u)
    k3 = dynamics(z + 0.5 * dt * k2, u)
    k4 = dynamics(z + dt * k3, u)
    zn = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    zn[1] = wrap_to_pi(zn[1])
    return zn


def pack(x, u):
    return np.concatenate([x.ravel(), u.ravel()])


def unpack(y, n=CFG.knots):
    x = y[: 4 * n].reshape(n, 4).copy()
    u = y[4 * n :].copy()
    return x, u


def initial_guess(cfg=CFG):
    n = cfg.knots
    t = np.linspace(0.0, cfg.horizon, n)
    tau = t / cfg.horizon
    smooth = 3.0 * tau**2 - 2.0 * tau**3

    x = np.zeros((n, 4))
    x[:, 0] = 0.9 * np.sin(2.0 * np.pi * tau)
    x[:, 1] = np.pi * (1.0 - smooth)
    x[:, 2] = np.gradient(x[:, 0], t)
    x[:, 3] = np.gradient(x[:, 1], t)
    x[0] = X0
    x[-1] = XF

    u = 6.0 * np.sin(2.0 * np.pi * tau[:-1])
    return pack(x, u)


def defect_constraints(y, cfg=CFG):
    x, u = unpack(y, cfg.knots)
    dt = cfg.horizon / (cfg.knots - 1)
    defects = []

    for k in range(cfg.knots - 1):
        fk = dynamics(x[k], u[k])
        fk1 = dynamics(x[k + 1], u[k])
        defect = x[k + 1] - x[k] - 0.5 * dt * (fk + fk1)
        defect[1] = wrap_to_pi(defect[1])
        defects.append(defect)

    return np.concatenate([x[0] - X0, x[-1] - XF, np.concatenate(defects)])


def objective(y, cfg=CFG):
    x, u = unpack(y, cfg.knots)
    dt = cfg.horizon / (cfg.knots - 1)
    theta_err = wrap_to_pi(x[:-1, 1])
    stage_cost = (
        cfg.q_x * x[:-1, 0] ** 2
        + cfg.q_theta * theta_err**2
        + cfg.q_v * (x[:-1, 2] ** 2 + x[:-1, 3] ** 2)
        + cfg.r_u * u**2
    )
    return float(dt * np.sum(stage_cost))


def solve_trajectory_optimization(cfg=CFG, p=P):
    n = cfg.knots
    y0 = initial_guess(cfg)

    state_lower = np.tile(np.array([-p.x_limit, -2.0 * np.pi, -20.0, -20.0]), n)
    state_upper = np.tile(np.array([p.x_limit, 2.0 * np.pi, 20.0, 20.0]), n)
    force_lower = np.full(n - 1, -p.u_limit)
    force_upper = np.full(n - 1, p.u_limit)
    bounds = Bounds(
        np.concatenate([state_lower, force_lower]),
        np.concatenate([state_upper, force_upper]),
    )

    constraints = [NonlinearConstraint(lambda y: defect_constraints(y, cfg), 0.0, 0.0)]

    result = minimize(
        lambda y: objective(y, cfg),
        y0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-7, "maxiter": 900, "disp": True},
    )

    x, u = unpack(result.x, n)
    metrics = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "max_defect": float(np.max(np.abs(defect_constraints(result.x.copy(), cfg)))),
        "max_abs_x": float(np.max(np.abs(x[:, 0]))),
        "max_abs_u": float(np.max(np.abs(u))),
        "x_limit": p.x_limit,
        "u_limit": p.u_limit,
    }
    return x, u, metrics


def linearize(z, u):
    eps = 1e-5
    a = np.zeros((4, 4))
    b = np.zeros((4, 1))

    for i in range(4):
        dz = np.zeros(4)
        dz[i] = eps
        a[:, i] = (dynamics(z + dz, u) - dynamics(z - dz, u)) / (2.0 * eps)

    b[:, 0] = (dynamics(z, u + eps) - dynamics(z, u - eps)) / (2.0 * eps)
    return a, b


def tvlqr(x_nom, u_nom, cfg=CFG, r_scale=1.0):
    dt = cfg.horizon / (cfg.knots - 1)
    q = np.diag([18.0, 45.0, 4.0, 4.0])
    qf = np.diag([120.0, 180.0, 25.0, 25.0])
    r = np.array([[0.25 * r_scale]])

    p_next = qf
    gains = []
    for k in reversed(range(cfg.knots - 1)):
        ac, bc = linearize(x_nom[k], u_nom[k])
        ad = np.eye(4) + dt * ac
        bd = dt * bc
        kk = -np.linalg.solve(r + bd.T @ p_next @ bd, bd.T @ p_next @ ad)
        p_next = q + ad.T @ p_next @ (ad + bd @ kk)
        gains.insert(0, kk)

    return gains


def terminal_lqr(r_scale=1.0):
    ac, bc = linearize(XF, 0.0)
    dt = CFG.horizon / (CFG.knots - 1)
    ad = np.eye(4) + dt * ac
    bd = dt * bc
    q = np.diag([20.0, 60.0, 5.0, 5.0])
    r = np.array([[0.25 * r_scale]])
    p = solve_discrete_are(ad, bd, q, r)
    return -np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)


def nominal_sample(x_nom, u_nom, t, cfg=CFG):
    dt = cfg.horizon / (cfg.knots - 1)
    k = min(int(t / dt), cfg.knots - 2)
    alpha = (t - k * dt) / dt
    x_ref = (1.0 - alpha) * x_nom[k] + alpha * x_nom[k + 1]
    x_ref[1] = wrap_to_pi(x_ref[1])
    return k, x_ref, float(u_nom[k])


def simulate_tvlqr(x_nom, u_nom, gains, cfg=CFG, r_scale=1.0, disturbance=None):
    dt = cfg.horizon / (cfg.knots - 1) / 5.0
    tf = cfg.horizon + 1.0
    steps = int(tf / dt) + 1

    t_hist = np.linspace(0.0, tf, steps)
    x_hist = np.zeros((steps, 4))
    u_hist = np.zeros(steps)

    z = X0.copy()
    if disturbance is not None:
        z = z + disturbance
        z[1] = wrap_to_pi(z[1])

    k_hold = terminal_lqr(r_scale)
    for i, t in enumerate(t_hist):
        if t <= cfg.horizon:
            k, x_ref, u_ref = nominal_sample(x_nom, u_nom, t, cfg)
            kk = gains[k]
        else:
            x_ref = XF
            u_ref = 0.0
            kk = k_hold

        err = z - x_ref
        err[1] = wrap_to_pi(err[1])
        u = float(u_ref + (kk @ err.reshape(-1, 1))[0, 0])
        u = float(np.clip(u, -P.u_limit, P.u_limit))

        x_hist[i] = z
        u_hist[i] = u
        z = rk4_step(z, u, dt)

    return t_hist, x_hist, u_hist


def plot_trajectory_and_tracking(t_nodes, x_nom, u_nom, t, x, u):
    FIG.mkdir(parents=True, exist_ok=True)

    theta_nom = continuous_angle(x_nom[:, 1], X0[1])
    theta_track = continuous_angle(x[:, 1], theta_nom[0])

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))

    ax[0].plot(t_nodes, x_nom[:, 0], label="optimized x")
    ax[0].plot(t, x[:, 0], "--", label="TVLQR tracked x")
    ax[0].axhline(P.x_limit, color="k", lw=0.8, ls=":")
    ax[0].axhline(-P.x_limit, color="k", lw=0.8, ls=":")
    ax[0].set_ylabel("cart x (m)")
    ax[0].legend()

    ax[1].plot(t_nodes, theta_nom, label="optimized theta")
    ax[1].plot(t, theta_track, "--", label="TVLQR tracked theta")
    ax[1].set_ylabel("theta (rad)")
    ax[1].legend()

    ax[2].step(t_nodes[:-1], u_nom, where="post", label="optimized force")
    ax[2].plot(t, u, "--", label="TVLQR force")
    ax[2].axhline(P.u_limit, color="k", lw=0.8, ls=":")
    ax[2].axhline(-P.u_limit, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("u (N)")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(FIG / "trajectory_and_tracking.png", dpi=180)
    plt.close(fig)


def plot_r_ablation(x_nom, u_nom, cfg=CFG):
    FIG.mkdir(parents=True, exist_ok=True)

    r_scales = [0.2, 1.0, 5.0]
    rows = []
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for r_scale in r_scales:
        gains = tvlqr(x_nom, u_nom, cfg, r_scale)
        t, x, u = simulate_tvlqr(
            x_nom,
            u_nom,
            gains,
            cfg,
            r_scale,
            disturbance=np.array([0.05, 0.08, 0.0, 0.0]),
        )
        terminal_mask = t >= cfg.horizon
        theta_err = np.array([wrap_to_pi(v) for v in x[:, 1]])
        theta_plot = continuous_angle(x[:, 1], X0[1])

        rows.append(
            {
                "R_scale": r_scale,
                "terminal_theta_rms_rad": float(np.sqrt(np.mean(theta_err[terminal_mask] ** 2))),
                "max_abs_force_N": float(np.max(np.abs(u))),
            }
        )
        ax[0].plot(t, theta_plot, label=f"R x {r_scale:g}")
        ax[1].plot(t, u, label=f"R x {r_scale:g}")

    ax[0].set_ylabel("theta (rad)")
    ax[0].legend()
    ax[1].set_ylabel("force (N)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(FIG / "lqr_R_ablation.png", dpi=180)
    plt.close(fig)
    return rows


def show_realtime_animation(t, x):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(-P.x_limit - 0.4, P.x_limit + 0.4)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.set_title(f"Cart-Pole TVLQR Real-Time Simulation - {STUDENT_NAME} {STUDENT_ID}")

    cart_w, cart_h = 0.34, 0.18
    cart = plt.Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h, fc="#4c78a8", ec="black")
    pole, = ax.plot([], [], lw=4, color="#f58518")
    bob, = ax.plot([], [], "o", ms=13, color="#e45756")
    time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)
    ax.add_patch(cart)

    dt = float(np.mean(np.diff(t)))
    stride = max(1, int(round(0.03 / dt)))
    frames = range(0, len(t), stride)
    interval_ms = max(1, int(round(1000.0 * dt * stride)))

    def update(i):
        cart_x = x[i, 0]
        theta = x[i, 1]
        tip_x = cart_x + P.l * math.sin(theta)
        tip_y = P.l * math.cos(theta)

        cart.set_xy((cart_x - cart_w / 2, -cart_h / 2))
        pole.set_data([cart_x, tip_x], [0.0, tip_y])
        bob.set_data([tip_x], [tip_y])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return cart, pole, bob, time_text

    animation = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=True, repeat=True)
    print("\nOpening real-time simulation window. Close the window to end the program.")
    plt.show()
    return animation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Constrained cart-pole trajectory optimization with TVLQR tracking."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate figures and metrics without opening the real-time animation window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Student: {STUDENT_NAME}, ID: {STUDENT_ID}")

    x_nom, u_nom, opt_metrics = solve_trajectory_optimization(CFG, P)
    t_nodes = np.linspace(0.0, CFG.horizon, CFG.knots)

    gains = tvlqr(x_nom, u_nom, CFG, r_scale=1.0)
    t, x_track, u_track = simulate_tvlqr(
        x_nom,
        u_nom,
        gains,
        CFG,
        r_scale=1.0,
        disturbance=np.array([0.05, 0.08, 0.0, 0.0]),
    )

    plot_trajectory_and_tracking(t_nodes, x_nom, u_nom, t, x_track, u_track)
    r_ablation = plot_r_ablation(x_nom, u_nom, CFG)

    metrics = {
        "student": {
            "name": STUDENT_NAME,
            "id": STUDENT_ID,
        },
        "trajectory_optimization": opt_metrics,
        "tracking": {
            "final_state": x_track[-1].tolist(),
            "max_abs_x": float(np.max(np.abs(x_track[:, 0]))),
            "max_abs_u": float(np.max(np.abs(u_track))),
        },
        "R_ablation": r_ablation,
        "generated_files": {
            "trajectory_plot": str(FIG / "trajectory_and_tracking.png"),
            "ablation_plot": str(FIG / "lqr_R_ablation.png"),
            "optimized_trajectory": str(OUT / "optimized_trajectory.npz"),
            "metrics": str(OUT / "metrics.json"),
            "real_time_animation": "shown in a Matplotlib window",
        },
    }

    np.savez(
        OUT / "optimized_trajectory.npz",
        t_nodes=t_nodes,
        x_nom=x_nom,
        u_nom=u_nom,
        t_tracking=t,
        x_tracking=x_track,
        u_tracking=u_track,
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== Optimization and Simulation Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not args.no_show:
        show_realtime_animation(t, x_track)


if __name__ == "__main__":
    main()
