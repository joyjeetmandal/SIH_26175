import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def render_sih_calibrated_mesh():
    # 📁 Path to the latest automated run outputs folder shown in your image
    RUN_DIR = r"outputs\runs\2026-09-10_15-19-03"
    
    npy_path = os.path.join(RUN_DIR, "heightmap.npy")
    meta_path = os.path.join(RUN_DIR, "metadata.json")
    
    print(f"📦 Loading SIH Core Data Assets from: {RUN_DIR}...")
    
    if not os.path.exists(npy_path):
        print(f"❌ Error: Could not find heightmap.npy at {npy_path}. Verify your folder paths.")
        return

    # 1. Read your teammate's compiled numerical elevation matrix
    height_grid = np.load(npy_path)
    
    # 2. Extract spatial properties dynamically from the generated matrix rows/cols
    rows, cols = height_grid.shape
    print(f"📐 Detected Matrix Resolution Grid: {cols} x {rows}")
    
    # Create coordinate grid indices matching the matrix bounds
    x_indices = np.arange(0, cols)
    y_indices = np.arange(0, rows)
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Flatten arrays for continuous 3D triangulation plotting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = height_grid.flatten()
    
    # 3. Read metadata metrics to pull ground-truth parameters (Min: 0.0m, Max: 20.03m)
    min_z, max_z = z_flat.min(), z_flat.max()
    mean_z = z_flat.mean()
    print(f"📊 Calibrated Topography Statistics -> Min: {min_z:.2f}m | Max: {max_z:.2f}m | Mean: {mean_z:.2f}m")

    print("🌌 Formatting 3D Spatial Canvas layout...")
    fig = plt.figure(figsize=(12, 9), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111116')
    
    # Set canvas background properties for an immersive dashboard feel
    ax.xaxis.set_pane_color((0.1, 0.1, 0.12, 1.0))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.12, 1.0))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.12, 1.0))
    
    # 4. Generate high-contrast continuous 3D Surface Terrain mapping
    print("📈 Generating metric-calibrated topographic surface terrain...")
    surf = ax.plot_trisurf(
        x_flat, y_flat, z_flat, 
        cmap='jet', 
        linewidth=0.02, 
        antialiased=True,
        vmin=0, vmax=21  # Bound the colormap limits to match the chat guidelines perfectly
    )
    
    # 5. Add color bar scale indicator matching physical meters
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=7, pad=0.08)
    cbar.set_label('True Elevation Height (Meters)', color='#ffffff', fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='#ffffff', labelcolor='#ffffff')
    
    # Customize axis labeling text styling
    ax.set_title("Smart India Hackathon - Calibrated 3D Terrain Analysis Engine", color='#ffffff', fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel("X Grid Pixel Units", color='#aaaaaa', fontsize=10)
    ax.set_ylabel("Y Grid Pixel Units", color='#aaaaaa', fontsize=10)
    ax.set_zlabel("Z Height Metric (m)", color='#aaaaaa', fontsize=10)
    
    # Change tick coordinate markers color profiling to blend with dark mode
    ax.tick_params(colors='#ffffff', labelsize=9)
    
    print("🚀 Launching interactive 3D window... (Click and drag to rotate your model live!)")
    plt.show()

if __name__ == "__main__":
    render_sih_calibrated_mesh()