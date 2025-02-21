import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def normalize_band(band):
    """Normalize a single band using percentile-based scaling"""
    valid = band[band > 0]  # Only consider non-zero values
    if len(valid) == 0:
        return np.zeros_like(band)
    
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    
    if p98 == p2:
        return np.zeros_like(band)
    
    return np.clip((band - p2) / (p98 - p2), 0, 1)

def visualize_sentinel2(data):
    """
    Visualize Sentinel-2 data with improved normalization
    """
    # Print data statistics
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.min():.3f} to {data.max():.3f}")
    print(f"Mean values per band:")
    for i in range(data.shape[2]):
        print(f"Band {i}: {data[:,:,i].mean():.3f}")
    
    # Create RGB composite (B4, B3, B2)
    rgb = data[:, :, [2, 1, 0]]  # Red, Green, Blue bands
    
    # Normalize each band separately
    rgb_norm = np.dstack([normalize_band(rgb[:,:,i]) for i in range(3)])
    
    # Create false color composite (NIR, SWIR, Red)
    false_color = data[:, :, [3, 4, 2]]  # NIR, SWIR1, Red bands
    false_color_norm = np.dstack([normalize_band(false_color[:,:,i]) for i in range(3)])
    
    # Plot with enhanced contrast
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(rgb_norm)
    ax1.set_title('RGB Composite')
    ax1.axis('off')
    
    ax2.imshow(false_color_norm)
    ax2.set_title('False Color Composite (NIR-SWIR-Red)')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_sentinel1(data):
    """
    Visualize Sentinel-1 SAR data
    Bands: VV and VH polarization
    """
    # Data should already be normalized from our processing
    vv = data[:, :, 0]
    vh = data[:, :, 1]
    
    # Create VV/VH ratio
    ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh!=0)
    ratio = np.clip(ratio, 0, 2)  # Clip ratio for better visualization
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    ax1.imshow(vv, cmap='gray')
    ax1.set_title('VV Polarization')
    ax1.axis('off')
    
    ax2.imshow(vh, cmap='gray')
    ax2.set_title('VH Polarization')
    ax2.axis('off')
    
    ax3.imshow(ratio, cmap='RdYlBu')
    ax3.set_title('VV/VH Ratio')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

def check_data(filename):
    """
    Debug function to check data content
    """
    data = np.load(filename)
    print(f"\nChecking {filename}:")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Range: {data.min()} to {data.max()}")
    print(f"Mean: {data.mean()}")
    print(f"Non-zero values: {np.count_nonzero(data)}")
    return data

def visualize_all_data(data_dir='khumbu_glacier_data'):
    """
    Visualize all downloaded Sentinel data
    """
    # Find all .npy files
    s1_files = sorted(glob.glob(os.path.join(data_dir, 's1_*.npy')))
    s2_files = sorted(glob.glob(os.path.join(data_dir, 's2_*.npy')))
    
    # Create output directory for plots
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Sentinel-1 data
    for file in s1_files:
        season = file.split('_')[1]
        data = np.load(file)
        fig = visualize_sentinel1(data)
        plt.savefig(os.path.join(output_dir, f'sentinel1_{season}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Process Sentinel-2 data
    for file in s2_files:
        season = file.split('_')[1]
        data = np.load(file)
        fig = visualize_sentinel2(data)
        plt.savefig(os.path.join(output_dir, f'sentinel2_{season}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    # visualize_all_data()
    # print("Visualization complete! Check the 'visualization_output' directory for results.")
    # s2_file = "khumbu_glacier_data/s2_winter_2024.npy"
    # data = check_data(s2_file)
    # fig = visualize_sentinel2(data)
    # plt.show()
    s2_files = glob.glob('khumbu_glacier_data/s2_*.npy')
    
    for file in s2_files:
        print(f"\nProcessing {file}:")
        data = np.load(file)
        
        # Convert to float32 if needed
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        
        fig = visualize_sentinel2(data)
        plt.savefig(f"{os.path.splitext(file)[0]}_viz.png", dpi=300, bbox_inches='tight')
        plt.close(fig)