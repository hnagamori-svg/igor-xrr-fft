"""
Generate sample XRR data for testing
"""

import numpy as np
import os

def generate_sample_xrr_data(
    thickness_layer1=6.0,  # nm
    thickness_layer2=7.0,  # nm
    critical_angle=0.6,    # deg
    noise_level=0.01,
    n_points=500,
    two_theta_max=8.0      # deg
):
    """
    Generate synthetic XRR data with Kiessig fringes
    """
    two_theta = np.linspace(0.1, two_theta_max, n_points)
    
    # Simplified XRR simulation with Kiessig fringes
    theta = two_theta / 2  # in degrees
    theta_rad = np.deg2rad(theta)
    crit_rad = np.deg2rad(critical_angle / 2)
    
    # Fresnel reflectivity
    qz = 4 * np.pi * np.sin(theta_rad) / 0.154  # A^-1
    
    # Base reflectivity (~1/theta^4)
    reflectivity = np.ones_like(two_theta)
    mask = two_theta > critical_angle
    reflectivity[mask] = (critical_angle / two_theta[mask])**4
    
    # Add Kiessig fringes
    total_thickness = thickness_layer1 + thickness_layer2
    fringe_period1 = 0.154 / (2 * thickness_layer1)  # approx period
    fringe_period2 = 0.154 / (2 * thickness_layer2)
    fringe_period_total = 0.154 / (2 * total_thickness)
    
    # Add oscillations
    osc1 = 0.2 * np.sin(2 * np.pi * qz * thickness_layer1 / 10)
    osc2 = 0.15 * np.sin(2 * np.pi * qz * thickness_layer2 / 10)
    osc_total = 0.25 * np.sin(2 * np.pi * qz * total_thickness / 10)
    
    # Modulate reflectivity with fringes
    reflectivity = reflectivity * (1 + osc1 + osc2 + osc_total)
    
    # Add noise
    reflectivity += noise_level * np.random.randn(n_points) * reflectivity
    
    # Scale to counts
    intensity = reflectivity * 1e6
    intensity = np.maximum(intensity, 1)  # Minimum 1 count
    
    return two_theta, intensity


def save_sample_data(filepath):
    """Save sample data to file"""
    two_theta, intensity = generate_sample_xrr_data()
    data = np.column_stack([two_theta, intensity])
    np.savetxt(filepath, data, header="2theta(deg) Intensity(counts)")
    print(f"Sample data saved to {filepath}")


if __name__ == "__main__":
    # Save to examples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(os.path.dirname(script_dir), "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    save_sample_data(os.path.join(examples_dir, "sample_xrr_data.txt"))
