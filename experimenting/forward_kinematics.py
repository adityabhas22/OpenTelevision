import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, FloatSlider
import ipywidgets as widgets

# --- 1. Math Functions (First Principles) ---

def get_rotation_matrix_z(theta_deg):
    """Creates a 3x3 rotation matrix for rotation around Z axis."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    # This matches the matrix derived in the text
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def get_homogeneous_transform(rotation_matrix, translation_vector):
    """Combines 3x3 Rotation and 3x1 Translation into 4x4 SE(3) Matrix."""
    T = np.eye(4) # Identity 4x4
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector
    return T

# --- 2. Visualization Logic ---

def plot_coordinate_frame(ax, T, length=1.0, label="Frame"):
    """
    Plots a 3D coordinate frame represented by 4x4 matrix T.
    Red = X, Green = Y, Blue = Z
    """
    # Extract Origin (The position vector P)
    origin = T[:3, 3]

    # Extract Basis Vectors (The columns of Rotation Matrix R)
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    # Plot X axis (Red)
    ax.quiver(origin[0], origin[1], origin[2], 
              x_axis[0], x_axis[1], x_axis[2], 
              color='r', length=length, normalize=True, arrow_length_ratio=0.1)
    
    # Plot Y axis (Green)
    ax.quiver(origin[0], origin[1], origin[2], 
              y_axis[0], y_axis[1], y_axis[2], 
              color='g', length=length, normalize=True, arrow_length_ratio=0.1)
    
    # Plot Z axis (Blue)
    ax.quiver(origin[0], origin[1], origin[2], 
              z_axis[0], z_axis[1], z_axis[2], 
              color='b', length=length, normalize=True, arrow_length_ratio=0.1)
    
    ax.text(origin[0], origin[1], origin[2], label, color='k')

# --- 3. The Interactive Loop ---

def visualize_transform(x, y, z, theta_z):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot World Frame (Identity) at 0,0,0
    T_world = np.eye(4)
    plot_coordinate_frame(ax, T_world, label="World {W}", length=1.5)
    
    # 2. Calculate New Frame {B}
    # Create rotation matrix (Rotated by theta around Z)
    R = get_rotation_matrix_z(theta_z)
    # Create translation vector
    P = np.array([x, y, z])
    # Combine into Homogeneous Transform
    T_body = get_homogeneous_transform(R, P)
    
    # 3. Plot New Frame
    plot_coordinate_frame(ax, T_body, label="Body {B}", length=1.0)
    
    # Visualize the connection (Translation vector)
    ax.plot([0, x], [0, y], [0, z], 'k--', alpha=0.5, label='Translation Vector P')

    # Formatting the plot
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(0, 5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.title(f"Homogeneous Transform Visualization\nRotation: {theta_z}° around Z")
    plt.show()

    # --- Print the Matrix to build intuition ---
    print("Homogeneous Transformation Matrix T:")
    # Formatting for clean printing
    with np.printoptions(precision=2, suppress=True):
        print(T_body)
    print("\nObserve:")
    print(f"1. The last column is exactly your position input: [{x}, {y}, {z}]")
    print(f"2. The top-left 3x3 is the Rotation Matrix.")
    print(f"3. Column 0 is the NEW X-axis direction.")
    print(f"4. Column 1 is the NEW Y-axis direction.")

# --- 4. Run the Widget ---
interact(visualize_transform, 
         x=FloatSlider(min=-3, max=3, step=0.1, value=2),
         y=FloatSlider(min=-3, max=3, step=0.1, value=2),
         z=FloatSlider(min=0, max=3, step=0.1, value=0),
         theta_z=FloatSlider(min=-180, max=180, step=5, value=45));