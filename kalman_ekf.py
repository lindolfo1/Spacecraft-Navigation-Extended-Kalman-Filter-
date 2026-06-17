import numpy as np 
from generate_orbits import make_orbit_state, compute_orbit, period_days, GM, step_size
from star_tracker import simulate_measurement, get_distance_from_sun, get_pos_from_sun, R_sun
from plot_kalman_ekf import plot_ekf, plot_comparison

dt = step_size


def init_state(init_orbit_state=[3.9, 0.6, 30.0, 45.0, 20.0], sensor_error=[1e-4, 1e-6, 1e-6]):
    if len(init_orbit_state) < 5:
        raise ValueError(f"Initial orbit parameters are not correctly defined (need to be length 5), received: ", init_orbit_state)
    if len(sensor_error) != 3:
        raise ValueError(f"Initial sensor error parameters are not correctly defined (need to be length 3), received: ", sensor_error)

    # generate orbit
    print("generating orbit...")
    s = make_orbit_state(a=init_orbit_state[0], e=init_orbit_state[1], inclination=init_orbit_state[2], lon_ascending_node=init_orbit_state[3], arg_periapsis=init_orbit_state[4])

    print(f"Period = {period_days(3.9):.1f} days")
    # Integrate for 2 full periods
    steps = int(2 * period_days(3.9))   # one step per day
    orbit_path = compute_orbit(s, num_steps=steps)     # [[x, y, z, vx, vy, vz]]
    
    # generate mock data
    # [r, θ, φ] r = distance from sun to satellite 
    sensor_data = []
    for o in orbit_path:
        z = simulate_measurement(o[:3], sensor_error)
        sensor_data.append(z)

    return sensor_data, orbit_path
    
def kalman_predict(x=[], P=[], F=[], Q=[]):
    """
    Performs the Kalman filter predict step.

    Parameters
    ----------
    x : list of float, length 6
        State vector [x, y, z, vx, vy, vz].
    P : list of list of float, shape (6, 6)
        State covariance matrix.
    F : list of list of float, shape (6, 6)
        State transition matrix.
    Q : list of list of float, shape (6, 6)
        Process noise covariance matrix.
    dt : float, optional
        Time step in seconds. Default is 86400 (1 day).

    Returns
    -------
    x : np.ndarray, shape (6,)
        Predicted state vector.
    P : np.ndarray, shape (6, 6)
        Predicted state covariance matrix.

    Raises
    ------
    ValueError
        If x is not length 6.
        If P is not shape (6, 6).
        If F is not shape (6, 6).
        If Q is not shape (6, 6).
    """

    pos = x[:3]
    vel = x[3:6]
    
    # compute gravity
    r = np.linalg.norm(pos)     # radius
    a = -(GM/r**3) * pos

    # propagate state
    pos_new = pos + vel*dt
    vel_new = vel +  a*dt
    x_new = np.concat([pos_new, vel_new])
    P_new = F @ P @ F.T + Q

    return x_new, P_new

def kalman_update(x=[], P=[], R=[], z=[]):
    """
    Performs the Kalman filter update step.

    Parameters
    ----------
    x : list of float, length 6
        State vector [x, y, z, vx, vy, vz].
    P : list of list of float, shape (6, 6)
        State covariance matrix.
    R : list of list of float, shape (3, 3)
        Measurement noise covariance matrix.
    z : list of float, length 3
        Measurement vector [solar_ang_rad, theta, phi].
    dt : float, optional
        Time step in seconds. Default is 86400 (1 day).

    Returns
    -------
    x : np.ndarray, shape (6,)
        Updated state vector.
    P : np.ndarray, shape (6, 6)
        Updated state covariance matrix.

    Raises
    ------
    ValueError
        If x is not length 6.
        If z is not length 3.
        If P is not shape (6, 6).
        If R is not shape (3, 3).
    """
    h = simulate_measurement(x[:3], [0, 0, 0])
    innovation = z-h

    H = build_H_jacobian(x)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_new = x + K @ innovation
    P_new = (np.eye(6) - K @ H) @ P

    return x_new, P_new


def build_H_jacobian(x=[]):
    H = np.zeros((3, 6))

    r_sq = x[0]**2 + x[1]**2 + x[2]**2
    rho = -R_sun / (r_sq * np.sqrt(r_sq - R_sun**2))
    H[0][0], H[0][1], H[0][2] = rho * x[0], rho * x[1], rho * x[2]
    H[1][0] = -x[1] / (x[0]**2 + x[1]**2)
    H[1][1] = x[0] / (x[0]**2 + x[1]**2)
    xy_sqrt = np.sqrt(x[0]**2 + x[1]**2)
    H[2][0] = -x[0]*x[2] / (r_sq * xy_sqrt)
    H[2][1] = -x[1]*x[2] / (r_sq * xy_sqrt)
    H[2][2] = 1 / xy_sqrt

    return H


def build_F(state=[]):
    x, y, z = state[0], state[1], state[2]
    r = np.linalg.norm(state[:3])

    F = np.eye(6)

    # x_n = x_(n-1) + v_(n-1) * dt (adding the dt term)
    F[0][3] = F[1][4] = F[2][5] = dt

    gm_dt = GM * dt
    r5 = r**5
    r2 = r**2
    x2 = x**2
    y2 = y**2
    z2 = z**2
    
    # ∂a/∂pos terms (gravity Jacobian); how gravity changes with position
    F[4][0] = F[3][1] = (3*gm_dt * x*y) / r5
    F[5][0] = F[3][2] = (3*gm_dt * x*z) / r5
    F[5][1] = F[4][2] = (3*gm_dt * y*z) / r5

    # cross terms 
    F[3][0] = (gm_dt * (3*x2 - r2)) / r5
    F[4][1] = (gm_dt * (3*y2 - r2)) / r5
    F[5][2] = (gm_dt * (3*z2 - r2)) / r5

    return F


def kalman_alg(sensor_data=[], sensor_error=[]):
    # get initial data 
    d0 = sensor_data[0]
    d1 = sensor_data[1]

    x0, y0, z0 = get_pos_from_sun(get_distance_from_sun(d0[0]), d0[1], d0[2])
    x1, y1, z1 = get_pos_from_sun(get_distance_from_sun(d1[0]), d1[1], d1[2])

    v1 = [(x1-x0)/dt, (y1-y0)/dt, (z1-z0)/dt]

    # array of all the states 
    states = []
    states.append(np.concatenate([[x1, y1, z1], v1]))

    # initialize all the matrices 
    P = [np.diag([1e-6, 1e-6, 1e-6,     # position AU²
             1e-8, 1e-8, 1e-8])]     # velocity (AU/s)²
    # velocity is harder to model
    Q = np.diag([1e-6, 1e-6, 1e-6,     # position AU²
             1e-8, 1e-8, 1e-8])     # velocity (AU/s)²
    sigma = np.array(sensor_error)**2
    R = np.diag(sigma)

    # len(sensor_data)-1 bc we need the 1st 2 samples for an init state 
    # sensor_data [0, 1, 2, 3, ..., n]
    # states         [0, 1, 2, ..., n-1]
    for i in range(len(sensor_data)-1):
        F = build_F(states[i])
        if i == 5:
            print(F)
        x_, P_ = kalman_predict(states[i], P[i], F, Q)
        x_new, P_new = kalman_update(x_, P_, R, sensor_data[i+1])
        states.append(x_new)
        P.append(P_new)

    return states, P


def naive_positioning(sensor_data=[]):
    measured_pos = []
    for pt in sensor_data:
        x, y, z = get_pos_from_sun(pt[0], pt[1], pt[2])
        measured_pos.append([x, y, z])
    return measured_pos
    



        



if __name__=="__main__":
    # [a, e, inclination, lon_ascending_node, arg_periapsis]
    init_orbit_state = [3.9, 0.6, 30.0, 45.0, 20.0]
    # [ρ, θ, φ]  ρ is more imprecise bc it is angular radius
    sensor_error = [1e-5, 1e-6, 1e-6]
    sensor_data, orbit_path = init_state(init_orbit_state, sensor_error)

    states, P = kalman_alg(sensor_data, sensor_error)

    naive = naive_positioning(sensor_data)

    plot_ekf(states, P, orbit_path, n_ellipses=20, output_path="ekf_plot-6.png")
    plot_comparison(states, naive, orbit_path, output_path="comparison_plot.png")

    