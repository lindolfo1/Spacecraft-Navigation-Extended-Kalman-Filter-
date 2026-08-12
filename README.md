# Spacecraft Navigation — Extended Kalman Filter

Autonomous deep-space navigation using an Extended Kalman Filter (EKF) fusing three passive optical measurements: solar angular radius and two star-tracker angles.

---

## Overview

This project implements an EKF to estimate a spacecraft's 3D position and velocity in heliocentric Cartesian coordinates `[x, y, z, vx, vy, vz]` using only passive optical measurements. The test orbit is a highly elliptical asteroid-belt trajectory (semi-major axis 3.9 AU, eccentricity 0.6, inclination 30°), which stresses the filter at aphelion where the sun appears smallest and range accuracy degrades sharply.

The filter is run for two full orbital periods (5626 daily steps) against a synthetic truth trajectory.

---

## Results

| Metric | Naive inversion | EKF |
|---|---|---|
| Position RMS | 0.1141 AU | **0.000182 AU** |
| Max position error | 0.653 AU | 0.0098 AU (initial transient) |
| Steady-state error | — | ~2 × 10⁻⁵ AU (~3000 km) |
| Mean NIS | — | 2.99 (target 3.0) |

**~627× RMS improvement** over treating each measurement independently.

### EKF vs true orbit

![EKF spacecraft navigation](ekf_plot.png)

The estimate converges within roughly 100 steps and holds a steady-state error near 2 × 10⁻⁵ AU. The 2σ uncertainty ellipses are drawn with an exaggeration factor (stated in the panel title) — at true scale they are sub-pixel against a 6 AU axis. They elongate along the sun line and grow toward aphelion, which is the expected geometry: range is the weakly observed direction, and it degrades as the solar disk shrinks.

### Naive positioning vs EKF

![Naive positioning vs EKF](comparison_plot.png)

Direct inversion of each measurement peaks near 0.65 AU at aphelion. Predicted 1σ range error is `r²·σ_ρ / R_sun`, giving ~0.167 AU at 6.24 AU — the observed peak is consistent with a ~4σ excursion. The EKF suppresses this by propagating orbital dynamics between measurements rather than trusting each reading on its own.

### Filter consistency

The bottom panel of the comparison figure plots normalised innovation squared (NIS) against the χ²₃ 95% band. Mean NIS of 2.99 against a target of 3.0 means the covariance is neither optimistic nor conservative — the filter's stated uncertainty matches its actual error statistics. This is the diagnostic worth checking first: a filter can produce a visually convincing trajectory while being badly inconsistent.

---

## Sensors

| Sensor | Measurement | Symbol | 1σ noise |
|---|---|---|---|
| Solar disk imager | Angular radius of the sun | ρ | 1 × 10⁻⁵ rad |
| Star tracker | Azimuth relative to background stars | θ | 1 × 10⁻⁶ rad |
| Star tracker | Elevation relative to background stars | φ | 1 × 10⁻⁶ rad |

Together `[ρ, θ, φ]` give full 3D position in spherical heliocentric coordinates. ρ is the least precise channel in practice because it must be inverted through `r = R_sun / sin ρ`, which amplifies angular error by `r²/R_sun`.

---

## Implementation notes

A few details that materially affect accuracy:

**State propagation uses RK4**, matching the truth integrator. Euler propagation (`pos + vel·dt`) drops the ½at² term and injects a one-sided bias of ~6 × 10⁻⁵ AU per step — far larger than the honest process noise. The covariance still propagates with the first-order transition matrix `F`, which is accurate enough over one day out of a ~2813 day period.

**Process noise is sized from an acceleration.** `build_Q` uses the piecewise white-noise-acceleration form driven by a single parameter `SIGMA_A`, including the position/velocity cross-covariance blocks. A diagonal `Q` discards the correlation between position and velocity error when both are driven by the same acceleration error.

**Angular innovations are wrapped to (−π, π].** θ comes from `arctan2` and this orbit encircles the origin, so without wrapping one step per revolution sees a ~2π innovation.

**Covariance update uses Joseph form** with explicit symmetrisation. Over 5626 steps the plain `(I − KH)P` accumulates asymmetry and can drift indefinite.

**Index alignment.** The first two measurements are consumed to seed position and a finite-difference velocity, so `states[i]` is the estimate at the epoch of `sensor_data[i+1]` and aligns with `orbit_path[i+1]`. Comparing against `orbit_path[:n]` instead introduces a one-day offset — 0.017 AU at perihelion, which is two orders of magnitude larger than the filter's actual error.

---

## Units

| Quantity | Unit |
|---|---|
| Position | AU |
| Velocity | AU / s |
| Time step | 86400 s (1 day) |
| GM | 3.964 × 10⁻¹⁴ AU³ s⁻² |
| R_sun | 0.002325 AU |
| Angles | radians |

---

## File structure

```
├── generate_orbits.py       # Keplerian orbit integrator (RK4), physical constants
├── star_tracker.py          # Sensor simulation, h(x), coordinate conversions
├── kalman_ekf.py            # EKF: predict, update, Jacobians F and H, Q sizing
├── plot_kalman_ekf.py       # Diagnostic plots: EKF, naive vs EKF, NIS consistency
├── noisy_data.py            # Standalone sensor-noise characterisation
├── kalman_test.py           # 1D Kalman toy problems (development scaffold)
├── phases_of_point_2.py     # Pulsar-phase fine positioning, cascade solver
├── fine_positioning.py      # Earlier single-shot pulsar solver (superseded)
├── ekf_plot.png
├── comparison_plot.png
├── kalman_filter_annotated-1.png   # Handwritten notes — algorithm and geometry
└── kalman_filter_annotated-2.png   # Handwritten notes — matrices and derivation
```

---

## Parameters

```python
# Orbit
a            = 3.9     # semi-major axis (AU)
e            = 0.6     # eccentricity
inclination  = 30.0    # degrees
lon_asc_node = 45.0    # degrees
arg_peri     = 20.0    # degrees

# Sensor noise (1σ, radians)
sigma_rho   = 1e-5     # solar angular radius
sigma_theta = 1e-6     # star tracker azimuth
sigma_phi   = 1e-6     # star tracker elevation

# Filter tuning
SIGMA_A = 1e-18        # unmodelled acceleration (AU/s²), drives Q
                       # for scale: solar gravity at 3.9 AU is ~2.6e-15 AU/s²

P0 = diag([1e-4]*3 +   # position variance (AU²), from the seeding range error
          [1e-13]*3)   # velocity variance ((AU/s)²), from finite differencing
```

`SIGMA_A` is the single most important knob. It was selected by sweeping 10⁻¹⁸ to 10⁻¹³ and choosing the value where mean NIS lands on 3.0. Larger values degrade accuracy monotonically (10⁻¹³ gives 0.033 AU RMS); the filter is not sensitive below ~10⁻¹⁷.

---

## Usage

```python
from generate_orbits import period_days
from kalman_ekf import init_state, kalman_alg, naive_positioning
from plot_kalman_ekf import plot_ekf, plot_comparison

init_orbit_state = [3.9, 0.6, 30.0, 45.0, 20.0]   # [a, e, inc, Ω, ω]
sensor_error     = [1e-5, 1e-6, 1e-6]             # [σ_ρ, σ_θ, σ_φ]

sensor_data, orbit_path = init_state(init_orbit_state, sensor_error)
states, P, nis = kalman_alg(sensor_data, sensor_error)

# states[i] aligns with orbit_path[i+1] — see Implementation notes
truth = orbit_path[1:]
naive = naive_positioning(sensor_data)[1:]

plot_ekf(states, P, truth, output_path="ekf_plot.png")
plot_comparison(states, naive, truth, nis=nis,
                period=period_days(3.9), output_path="comparison_plot.png")
```

Or just run `python kalman_ekf.py`, which prints the RMS summary and writes both figures.

---

## Dependencies

```
numpy
matplotlib
```

---

## Limitations

**The dynamics model is exact.** The truth trajectory and the filter's prediction step use the same RK4 integrator, the same GM, and the same step size, so there is no genuine process noise for the filter to contend with. A real mission faces mismodelled solar radiation pressure, third-body perturbations, and unmodelled thrusting. The 627× figure should be read as an upper bound on what the dynamics model can buy you, not a mission-representative number.

**`SIGMA_A` is tuned on the run it is reported on.** A defensible benchmark would tune on one noise realisation and report on an independent one.

**Range is the weak direction.** `dr/dρ = −R_sun / sin²ρ` grows as `r²/R_sun`, so radial uncertainty scales quadratically with heliocentric distance. This is a real observability limit of the sensor suite and shows up clearly in the orientation of the covariance ellipses — the EKF mitigates it via orbital dynamics but cannot remove it.

**No measurement outlier handling.** There is no innovation gating or robust update, so a single bad reading propagates directly into the state.

**Single-target, no attitude estimation.** The star tracker is assumed to deliver sun-relative angles directly; real attitude determination and the associated error budget are out of scope.
