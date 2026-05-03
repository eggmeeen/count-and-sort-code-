# Project 3: Constrained Cart-Pole Trajectory Optimization

Student: 李皓宁  
Student ID: 2023112386

This project implements constrained trajectory optimization and TVLQR tracking for a cart-pole system. The task is to swing the pole from the hanging equilibrium to the upright equilibrium while satisfying hard constraints on cart position and motor force.

## What The Code Does

- Builds the nonlinear cart-pole dynamics from the Lagrange equations.
- Solves a direct-transcription trajectory optimization problem with `scipy.optimize.SLSQP`.
- Enforces cart position and motor force limits:
  - `|x| <= 2.4 m`
  - `|u| <= 30 N`
- Designs a time-varying LQR controller along the optimized trajectory.
- Simulates TVLQR tracking with an initial disturbance.
- Saves trajectory/tracking plots and prints optimization metrics.
- Opens a real-time Matplotlib animation window by default.

## Files

- `simple.py`: main Python implementation.
- `outputs/figures/trajectory_and_tracking.png`: generated trajectory and TVLQR tracking plot.
- `outputs/figures/lqr_R_ablation.png`: generated LQR `R`-matrix ablation plot.
- `outputs/metrics.json`: generated numerical metrics.
- `outputs/optimized_trajectory.npz`: generated optimized trajectory and tracking data.

The `outputs/` directory is generated when the script runs and is not required before execution.

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

## Run

Run with real-time animation:

```bash
python3 simple.py
```

Run without opening the animation window:

```bash
python3 simple.py --no-show
```

## Expected Console Output

The script prints the student information, SLSQP optimization status, trajectory optimization metrics, TVLQR tracking metrics, and `R`-matrix ablation results.

Typical successful metrics are:

- optimization success: `True`
- maximum dynamics defect: about `1e-9`
- maximum cart displacement: about `0.7 m`, below the `2.4 m` limit
- maximum force: about `14 N`, below the `30 N` limit

## Method Summary

The state is

```text
z = [x, theta, x_dot, theta_dot]
```

where `theta = 0` is the upright equilibrium and `theta = pi` is the hanging initial pose.

The direct-transcription problem discretizes a 4-second trajectory into 51 knot points and uses trapezoidal dynamics defects:

```text
z[k+1] - z[k] - dt/2 * (f(z[k], u[k]) + f(z[k+1], u[k])) = 0
```

After obtaining the open-loop trajectory, the code numerically linearizes the dynamics along the trajectory and solves the finite-horizon Riccati recursion for TVLQR tracking:

```text
u = u_ref[k] + K[k] (z - z_ref[k])
```

The final part compares several LQR control penalties by scaling the `R` matrix.
