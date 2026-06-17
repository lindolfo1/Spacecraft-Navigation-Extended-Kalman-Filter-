import numpy as np 
import pandas as pd
import random

# all calculations measuring angular radius not angular diameter bc as you get close to the sun, the apparent angular diameter changes bc you are not seeing the sun at infinity and cannot see the poles 
R_sun = 0.002325235       # in AU

def generate_noisy_measurements(true_value=0, noise_std=1.0, n=100, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    measurements = true_value + noise
    return measurements, true_value

def add_noise(value=0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std)
    return float(value + noise)

# coordinates in AU 
def simulate_angular_radius(x, y, z, noise_std=0.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(R_sun/r)
    return add_noise(theta, noise_std)


def simulate_sun_direction(x, y, z, theta_noise_std=0.0, phi_noise_std=0.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    ux, uy, uz = x/r, y/r, z/r
    phi = np.arcsin(uz)
    theta = np.arctan2(uy, ux)
    return add_noise(theta, theta_noise_std), add_noise(phi, phi_noise_std)


def simulate_measurement(x, noise_std=[0.0, 0.0, 0.0]):
    px, py, pz = x[0], x[1], x[2]
    r = np.linalg.norm([px, py, pz])
    rho   = np.arcsin(R_sun / r)
    theta = np.arctan2(py, px)
    phi   = np.arcsin(pz / r)
    if noise_std[0] > 0:
        rho = add_noise(rho, noise_std[0])
    if noise_std[1] > 0.0:
        theta = add_noise(theta, noise_std[1])
    if noise_std[2] > 0.0:
        phi = add_noise(phi, noise_std[2])
    return np.array([rho, theta, phi])
    

    

def get_distance_from_sun(apparent_angular_radius):
    return float(R_sun / np.sin(apparent_angular_radius))


def get_pos_from_sun(r, theta, phi):
    # phi = elevation from XY plane (arcsin convention)
    # theta = azimuthal angle in XY plane (arctan2 convention)
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return float(x), float(y), float(z)




if __name__ == "__main__":
    x, y, z = 3.9, 3.4, 0.07
    print(f"actual dist: ", np.sqrt(x**2 + y**2 + z**2))
    ang_r = simulate_angular_radius(x, y, z, error=0.0)
    theta, phi = simulate_sun_direction(x, y, z)
    print(f"(ang_r, theta, phi): ({ang_r:0.10f}, {theta:10.10f}, {phi:10.10f})")

    ap_r = get_distance_from_sun(ang_r)
    print(f"apparent distance: ", ap_r)
    ap_x, ap_y, ap_z = get_pos_from_sun(ap_r, theta, phi)
    print(f"(ap_x, ap_y, ap_z): ({ap_x:0.4f}, {ap_y:6.4f}, {ap_z:6.4f})")

