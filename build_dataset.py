import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection,
    MimeType, bbox_to_dimensions
)
import cv2
from sklearn.cluster import KMeans
from config import SH_CLIENT_ID, SH_CLIENT_SECRET

def setup_sentinel_hub():
    """Setup Sentinel Hub configuration with explicit credentials"""
    config = SHConfig()
    
    # Set credentials from config file
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET
    
    # Validate configuration
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Sentinel Hub credentials not properly configured. "
            "Please check your config.py file."
        )
    
    # Simple validation without making an actual data request
    print("Authentication configured with:")
    print(f"  Client ID: {config.sh_client_id[:5]}...")
    print(f"  Client Secret: {config.sh_client_secret[:3]}...")
    
    return config

def get_khumbu_bbox():
    """Define bounding box for Khumbu Glacier"""
    return BBox(
        bbox=[86.78, 27.95, 86.85, 28.00],
        crs=CRS.WGS84
    )

def download_khumbu_temporal_data(years, resolution=10):
    """
    Download multi-temporal data for Khumbu Glacier
    
    Parameters:
    - years: list of years to download data for
    - resolution: spatial resolution in meters
    """
    config = setup_sentinel_hub()
    bbox = get_khumbu_bbox()
    
    # Calculate dimensions
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"Image dimensions for Khumbu: {bbox_size} pixels")
    
    # Create output directories
    data_dir = "khumbu_glacier_data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/optical", exist_ok=True)
    os.makedirs(f"{data_dir}/sar", exist_ok=True)
    os.makedirs(f"{data_dir}/ndsi", exist_ok=True)
    os.makedirs(f"{data_dir}/ndwi", exist_ok=True)
    
    # Define summer periods for each year (best for glacier mapping)
    for year in years:
        # Summer period (minimal snow cover)
        summer_start = f"{year}-07-01"
        summer_end = f"{year}-09-30"
        time_interval = (summer_start, summer_end)
        
        try:
            # 1. Download optical data (Sentinel-2)
            print(f"Requesting optical data for Khumbu {year}...")
            optical_request = SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
                                units: "REFLECTANCE"
                            }],
                            output: {
                                bands: 7,
                                sampleType: "FLOAT32"
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        return [sample.B02, sample.B03, sample.B04, sample.B05, 
                                sample.B08, sample.B11, sample.B12];
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC' # Least cloud coverage
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_size,
                config=config
            )

            # Execute optical request
            optical_data = optical_request.get_data()
            if len(optical_data) > 0:
                np.save(f"{data_dir}/optical/khumbu_{year}_optical.npy", optical_data[0])
                print(f"Downloaded optical data for Khumbu {year}")
                
                # Display a preview of the data
                # Be sure to clip values to 0-1 range for visualization
                rgb = np.clip(optical_data[0][:, :, :3], 0, 1)  # First 3 bands are RGB
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb)
                plt.title(f"Khumbu Glacier Sentinel-2 RGB Preview ({year})")
                plt.axis('off')
                plt.savefig(f"{data_dir}/optical/khumbu_{year}_rgb_preview.png")
                plt.close()
            else:
                print(f"No optical data available for Khumbu {year}")
            
            # 2. Download SAR data (Sentinel-1)
            print(f"Requesting SAR data for Khumbu {year}...")
            sar_request = SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["VV", "VH"],
                                units: "LINEAR_POWER"
                            }],
                            output: {
                                bands: 2,
                                sampleType: "FLOAT32"
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        // Convert to dB scale and normalize
                        let VVdB = 10 * Math.log10(sample.VV);
                        let VHdB = 10 * Math.log10(sample.VH);
                        
                        // Normalize to 0-1 range (typical SAR values are between -25 and 0 dB)
                        let VVnorm = (VVdB + 25) / 25;
                        let VHnorm = (VHdB + 25) / 25;
                        
                        return [VVnorm, VHnorm];
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL1_IW,
                        time_interval=time_interval,
                        other_args={'orthorectify': True}
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_size,
                config=config
            )
            
            # Execute SAR request
            sar_data = sar_request.get_data()
            if len(sar_data) > 0:
                np.save(f"{data_dir}/sar/khumbu_{year}_sar.npy", sar_data[0])
                print(f"Downloaded SAR data for Khumbu {year}")
                
                # Display a preview of the data
                vv = sar_data[0][:, :, 0]  # VV band
                plt.figure(figsize=(10, 8))
                plt.imshow(vv, cmap='gray')
                plt.title(f"Khumbu Glacier Sentinel-1 SAR Preview ({year})")
                plt.axis('off')
                plt.savefig(f"{data_dir}/sar/khumbu_{year}_sar_preview.png")
                plt.close()
            else:
                print(f"No SAR data available for Khumbu {year}")
            
            # 3. Download NDSI (snow index)
            print(f"Requesting NDSI data for Khumbu {year}...")
            ndsi_request = SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["B03", "B11"],
                                units: "REFLECTANCE"
                            }],
                            output: {
                                bands: 1,
                                sampleType: "FLOAT32"
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        // Calculate NDSI (Normalized Difference Snow Index)
                        let ndsi = (sample.B03 - sample.B11) / (sample.B03 + sample.B11);
                        return [ndsi];
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_size,
                config=config
            )
            
            # Execute NDSI request
            ndsi_data = ndsi_request.get_data()
            if len(ndsi_data) > 0:
                # Check if the data is 2D or 3D
                ndsi_array = ndsi_data[0]
                
                # If 2D, reshape to 3D (height, width, 1)
                if len(ndsi_array.shape) == 2:
                    ndsi_array = ndsi_array.reshape(ndsi_array.shape[0], ndsi_array.shape[1], 1)
                
                np.save(f"{data_dir}/ndsi/khumbu_{year}_ndsi.npy", ndsi_array)
                print(f"Downloaded NDSI data for Khumbu {year}")
                
                # Display a preview of the data - be careful about dimensions
                ndsi_display = ndsi_array[:, :, 0] if len(ndsi_array.shape) == 3 else ndsi_array
                plt.figure(figsize=(10, 8))
                plt.imshow(ndsi_display, cmap='RdBu', vmin=-1, vmax=1)
                plt.colorbar(label='NDSI')
                plt.title(f"Khumbu Glacier NDSI Preview ({year})")
                plt.axis('off')
                plt.savefig(f"{data_dir}/ndsi/khumbu_{year}_ndsi_preview.png")
                plt.close()
            else:
                print(f"No NDSI data available for Khumbu {year}")
            
            # 4. Download NDWI (water index for supraglacial lakes)
            print(f"Requesting NDWI data for Khumbu {year}...")
            ndwi_request = SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["B03", "B08"],
                                units: "REFLECTANCE"
                            }],
                            output: {
                                bands: 1,
                                sampleType: "FLOAT32"
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        // Calculate NDWI (Normalized Difference Water Index)
                        let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
                        return [ndwi];
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_size,
                config=config
            )
            
            # Execute NDWI request
            ndwi_data = ndwi_request.get_data()
            if len(ndwi_data) > 0:
                # Check if the data is 2D or 3D
                ndwi_array = ndwi_data[0]
                
                # If 2D, reshape to 3D (height, width, 1)
                if len(ndwi_array.shape) == 2:
                    ndwi_array = ndwi_array.reshape(ndwi_array.shape[0], ndwi_array.shape[1], 1)
                
                np.save(f"{data_dir}/ndwi/khumbu_{year}_ndwi.npy", ndwi_array)
                print(f"Downloaded NDWI data for Khumbu {year}")
                
                # Display a preview of the data - be careful about dimensions
                ndwi_display = ndwi_array[:, :, 0] if len(ndwi_array.shape) == 3 else ndwi_array
                plt.figure(figsize=(10, 8))
                plt.imshow(ndwi_display, cmap='RdYlBu', vmin=-1, vmax=1)
                plt.colorbar(label='NDWI')
                plt.title(f"Khumbu Glacier NDWI Preview ({year})")
                plt.axis('off')
                plt.savefig(f"{data_dir}/ndwi/khumbu_{year}_ndwi_preview.png")
                plt.close()
            else:
                print(f"No NDWI data available for Khumbu {year}")
                
                # In case no NDWI data is available, create an empty one based on the NDSI shape
                # This ensures that the pipeline doesn't break
                if os.path.exists(f"{data_dir}/ndsi/khumbu_{year}_ndsi.npy"):
                    ndsi_array = np.load(f"{data_dir}/ndsi/khumbu_{year}_ndsi.npy")
                    # Create empty NDWI with same shape as NDSI
                    empty_ndwi = np.zeros_like(ndsi_array)
                    np.save(f"{data_dir}/ndwi/khumbu_{year}_ndwi.npy", empty_ndwi)
                    print(f"Created empty NDWI data for Khumbu {year} based on NDSI shape")
            
        except Exception as e:
            print(f"Error downloading data for Khumbu {year}: {str(e)}")
            print("Continuing with next year...")
    
    return data_dir

def create_synthetic_sentinel_data():
    """
    Create synthetic Sentinel-1 and Sentinel-2 data for Khumbu Glacier
    Used as a fallback when SentinelHub downloads fail
    """
    data_dir = "khumbu_glacier_data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/optical", exist_ok=True)
    os.makedirs(f"{data_dir}/sar", exist_ok=True)
    os.makedirs(f"{data_dir}/ndsi", exist_ok=True)
    os.makedirs(f"{data_dir}/ndwi", exist_ok=True)
    
    # Years to simulate
    years = [2018, 2019, 2020, 2021, 2022]
    
    # Image dimensions - similar to Sentinel-2 at 10m resolution for the Khumbu AOI
    height, width = 689, 553
    
    print(f"Creating synthetic Sentinel data with dimensions: {width}x{height} pixels")
    
    # Create base glacier structure
    # Initialize with random elevation model for the terrain
    dem = np.zeros((height, width))
    
    # Create mountain-like terrain with peaks
    for _ in range(5):
        peak_y = np.random.randint(height//4, 3*height//4)
        peak_x = np.random.randint(width//4, 3*width//4)
        peak_height = np.random.uniform(0.7, 1.0)
        peak_width = np.random.uniform(width//3, width//2)
        
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((y - peak_y)**2 + (x - peak_x)**2)
        dem += peak_height * np.exp(-dist**2 / (2 * peak_width**2))
    
    # Normalize DEM to 0-1 range
    dem = (dem - dem.min()) / (dem.max() - dem.min())
    
    # Create glacier path (flowing from high elevation to low)
    glacier_center_y = height // 2
    glacier_center_x = width // 2
    glacier_width = width // 3
    
    # Create mask for where glacier can exist (where elevation is high enough)
    elevation_threshold = 0.6
    high_elevation = dem > elevation_threshold
    
    # Create glacier core in high elevation areas
    # This will be our baseline for year 1
    glacier_core = np.zeros((height, width), dtype=bool)
    y, x = np.ogrid[:height, :width]
    
    # Create a meandering glacier path
    for i in range(8):
        center_x = glacier_center_x + int(np.sin(i * 0.8) * glacier_width * 0.3)
        center_y = glacier_center_y + i * height // 10
        if center_y < height:
            radius = glacier_width // 2 - i * 5
            if radius > 10:
                glacier_core |= ((y - center_y)**2 + (x - center_x)**2) < radius**2
    
    # Combine with elevation to make it look natural
    glacier_core &= dem > 0.4  # Only in medium-high elevations
    
    for year_idx, year in enumerate(years):
        print(f"Creating data for year {year}...")
        
        # Glaciers retreat over time - simulate this by eroding edges
        erosion_kernel = np.ones((3, 3), np.uint8)
        glacier_mask = glacier_core.copy()
        
        # Apply more erosion for later years to simulate retreat
        for i in range(year_idx * 2):
            glacier_mask = cv2.erode(glacier_mask.astype(np.uint8), erosion_kernel).astype(bool)
        
        # Add some random noise to boundaries for realism
        boundary = cv2.dilate(glacier_mask.astype(np.uint8), erosion_kernel) - glacier_mask.astype(np.uint8)
        boundary_mask = boundary > 0
        
        # Randomly remove some boundary pixels
        random_boundary = np.random.rand(height, width) > 0.6
        glacier_mask = glacier_mask | (boundary_mask & random_boundary)
        
        # Create debris-covered areas on lower parts of glacier
        debris_mask = np.zeros_like(glacier_mask)
        lower_glacier = (glacier_mask) & (dem < 0.65)  # Debris tends to be in lower elevations
        debris_mask |= lower_glacier
        
        # Add randomness to debris distribution
        random_debris = np.random.rand(height, width) > 0.7
        debris_mask &= random_debris
        
        # Create clean ice mask (glacier - debris)
        clean_ice_mask = glacier_mask & (~debris_mask)
        
        # Create Sentinel-2 optical bands
        # 8 bands: B02, B03, B04, B05, B08, B11, B12, SCL
        optical = np.zeros((height, width, 8))
        
        # Create base land cover with some variation
        land = np.random.normal(0.3, 0.05, (height, width)) * (dem < 0.5)  # Low-elevation terrain
        vegetation = np.random.normal(0.15, 0.05, (height, width)) * ((dem > 0.3) & (dem < 0.6))  # Mid-elevation vegetation
        rocks = np.random.normal(0.25, 0.05, (height, width)) * ((dem > 0.5) & (~glacier_mask))  # High-elevation rocks
        water = np.random.normal(0.1, 0.02, (height, width)) * (dem < 0.2)  # Low areas have water
        
        # Add values for each band with realistic spectral signatures
        # Blue (B02)
        optical[:,:,0] = land * 0.3 + vegetation * 0.2 + rocks * 0.25 + water * 0.4
        optical[:,:,0][clean_ice_mask] = np.random.normal(0.8, 0.1, np.sum(clean_ice_mask))  # Clean ice is bright
        optical[:,:,0][debris_mask] = np.random.normal(0.4, 0.1, np.sum(debris_mask))  # Debris is darker
        
        # Green (B03)
        optical[:,:,1] = land * 0.4 + vegetation * 0.5 + rocks * 0.3 + water * 0.3
        optical[:,:,1][clean_ice_mask] = np.random.normal(0.75, 0.1, np.sum(clean_ice_mask))
        optical[:,:,1][debris_mask] = np.random.normal(0.45, 0.1, np.sum(debris_mask))
        
        # Red (B04)
        optical[:,:,2] = land * 0.5 + vegetation * 0.2 + rocks * 0.35 + water * 0.2
        optical[:,:,2][clean_ice_mask] = np.random.normal(0.7, 0.1, np.sum(clean_ice_mask))
        optical[:,:,2][debris_mask] = np.random.normal(0.5, 0.1, np.sum(debris_mask))
        
        # Red Edge (B05)
        optical[:,:,3] = land * 0.6 + vegetation * 0.7 + rocks * 0.4 + water * 0.1
        optical[:,:,3][clean_ice_mask] = np.random.normal(0.6, 0.1, np.sum(clean_ice_mask))
        optical[:,:,3][debris_mask] = np.random.normal(0.55, 0.1, np.sum(debris_mask))
        
        # NIR (B08)
        optical[:,:,4] = land * 0.5 + vegetation * 0.8 + rocks * 0.45 + water * 0.05
        optical[:,:,4][clean_ice_mask] = np.random.normal(0.4, 0.1, np.sum(clean_ice_mask))  # Ice absorbs NIR
        optical[:,:,4][debris_mask] = np.random.normal(0.5, 0.1, np.sum(debris_mask))
        
        # SWIR (B11)
        optical[:,:,5] = land * 0.6 + vegetation * 0.4 + rocks * 0.5 + water * 0.01
        optical[:,:,5][clean_ice_mask] = np.random.normal(0.1, 0.05, np.sum(clean_ice_mask))  # Ice is very dark in SWIR
        optical[:,:,5][debris_mask] = np.random.normal(0.4, 0.1, np.sum(debris_mask))
        
        # SWIR (B12)
        optical[:,:,6] = land * 0.65 + vegetation * 0.35 + rocks * 0.55 + water * 0.01
        optical[:,:,6][clean_ice_mask] = np.random.normal(0.08, 0.04, np.sum(clean_ice_mask))  # Ice is very dark in SWIR
        optical[:,:,6][debris_mask] = np.random.normal(0.45, 0.1, np.sum(debris_mask))
        
        # SCL (Scene Classification Layer)
        optical[:,:,7] = np.ones((height, width)) * 4  # Default vegetation
        optical[:,:,7][water > 0.05] = 6  # Water
        optical[:,:,7][clean_ice_mask] = 11  # Snow/ice value in SCL
        optical[:,:,7][debris_mask] = 5  # Assuming debris might be classified as bare soil
        optical[:,:,7][rocks > 0.2] = 5  # Bare soil/rocks
        
        # Create Sentinel-1 SAR data (VV, VH)
        sar = np.zeros((height, width, 2))
        
        # VV band - vertical send, vertical receive
        sar[:,:,0] = land * 0.4 + vegetation * 0.5 + rocks * 0.6 + water * 0.1
        sar[:,:,0][clean_ice_mask] = np.random.normal(0.3, 0.07, np.sum(clean_ice_mask))  # Ice has medium-low backscatter
        sar[:,:,0][debris_mask] = np.random.normal(0.6, 0.1, np.sum(debris_mask))  # Debris has higher backscatter
        
        # VH band - vertical send, horizontal receive
        sar[:,:,1] = land * 0.25 + vegetation * 0.4 + rocks * 0.35 + water * 0.05
        sar[:,:,1][clean_ice_mask] = np.random.normal(0.2, 0.05, np.sum(clean_ice_mask))  # Lower cross-pol backscatter
        sar[:,:,1][debris_mask] = np.random.normal(0.4, 0.1, np.sum(debris_mask))
        
        # Calculate NDSI (Normalized Difference Snow Index)
        # NDSI = (Green - SWIR1) / (Green + SWIR1)
        green = optical[:,:,1]
        swir1 = optical[:,:,5]
        ndsi_values = np.zeros((height, width))
        valid_pixels = (green + swir1) != 0
        ndsi_values[valid_pixels] = (green[valid_pixels] - swir1[valid_pixels]) / (green[valid_pixels] + swir1[valid_pixels])
        
        # Add realistic values
        ndsi_values[clean_ice_mask] = np.clip(np.random.normal(0.7, 0.1, np.sum(clean_ice_mask)), 0, 1)  # Snow/ice has high NDSI
        ndsi_values[debris_mask] = np.clip(np.random.normal(0.2, 0.1, np.sum(debris_mask)), -1, 1)  # Debris has lower NDSI
        
        # Create NDWI (Normalized Difference Water Index)
        # NDWI = (Green - NIR) / (Green + NIR)
        green = optical[:,:,1]
        nir = optical[:,:,4]
        ndwi_values = np.zeros((height, width))
        valid_pixels = (green + nir) != 0
        ndwi_values[valid_pixels] = (green[valid_pixels] - nir[valid_pixels]) / (green[valid_pixels] + nir[valid_pixels])
        
        # Add realistic values for water bodies
        ndwi_values[water > 0.05] = np.clip(np.random.normal(0.6, 0.1, np.sum(water > 0.05)), 0, 1)
        
        # Reshape to match expected dimensions
        ndsi = ndsi_values.reshape(height, width, 1)
        ndwi = ndwi_values.reshape(height, width, 1)
        
        # Save the data
        np.save(f"{data_dir}/optical/khumbu_{year}_optical.npy", optical)
        np.save(f"{data_dir}/sar/khumbu_{year}_sar.npy", sar)
        np.save(f"{data_dir}/ndsi/khumbu_{year}_ndsi.npy", ndsi)
        np.save(f"{data_dir}/ndwi/khumbu_{year}_ndwi.npy", ndwi)
        
        # Also save masks for reference
        np.save(f"{data_dir}/optical/khumbu_{year}_glacier_mask.npy", glacier_mask.astype(np.uint8))
        np.save(f"{data_dir}/optical/khumbu_{year}_debris_mask.npy", debris_mask.astype(np.uint8))
        
        # Visualize the created data
        rgb = optical[:,:,:3]  # RGB bands
        rgb = np.clip(rgb, 0, 1)  # Ensure valid RGB range
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(rgb)
        plt.title(f"RGB Composite {year}")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(glacier_mask, cmap='Blues')
        plt.title(f"Glacier Extent {year}")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(debris_mask, cmap='Oranges')
        plt.title(f"Debris Cover {year}")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(ndsi_values, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"NDSI {year}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{data_dir}/optical/khumbu_{year}_preview.png")
        plt.close()
    
    print(f"Created synthetic multi-temporal Sentinel data for {len(years)} years")
    return data_dir

def generate_khumbu_change_masks(years):
    """
    Generate change detection masks for Khumbu Glacier
    This function creates labels for glacier change between consecutive years
    """
    data_dir = "khumbu_glacier_data"
    os.makedirs(f"{data_dir}/change_masks", exist_ok=True)
    
    # Process data for years
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        try:
            # Load optical data
            optical1_path = f"{data_dir}/optical/khumbu_{year1}_optical.npy"
            optical2_path = f"{data_dir}/optical/khumbu_{year2}_optical.npy"
            
            # Load NDSI data
            ndsi1_path = f"{data_dir}/ndsi/khumbu_{year1}_ndsi.npy"
            ndsi2_path = f"{data_dir}/ndsi/khumbu_{year2}_ndsi.npy"
            
            # Load SAR data
            sar1_path = f"{data_dir}/sar/khumbu_{year1}_sar.npy"
            sar2_path = f"{data_dir}/sar/khumbu_{year2}_sar.npy"
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [optical1_path, optical2_path, ndsi1_path, ndsi2_path, sar1_path, sar2_path]):
                print(f"Missing data for change detection between {year1} and {year2}")
                continue
            
            # Load data
            optical1 = np.load(optical1_path)
            optical2 = np.load(optical2_path)
            ndsi1 = np.load(ndsi1_path)
            ndsi2 = np.load(ndsi2_path)
            sar1 = np.load(sar1_path)
            sar2 = np.load(sar2_path)
            
            # Ensure NDSI is 2D for thresholding
            if len(ndsi1.shape) == 3:
                ndsi1_2d = ndsi1[:, :, 0]  # Extract the first channel
            else:
                ndsi1_2d = ndsi1
                
            if len(ndsi2.shape) == 3:
                ndsi2_2d = ndsi2[:, :, 0]
            else:
                ndsi2_2d = ndsi2
            
            # 1. Generate glacier masks using NDSI
            # NDSI > 0.4 typically indicates snow/ice
            glacier_mask1 = (ndsi1_2d > 0.4).astype(np.uint8)
            glacier_mask2 = (ndsi2_2d > 0.4).astype(np.uint8)
            
            # 2. Detect debris-covered areas using optical and SAR data
            # Extract visible bands (RGB) from optical data
            rgb1 = optical1[:, :, :3]
            rgb2 = optical2[:, :, :3]
            
            # Use k-means clustering to identify potential debris-covered areas
            # Reshape for clustering
            pixels1 = rgb1.reshape(-1, 3)
            pixels2 = rgb2.reshape(-1, 3)
            
            # Apply K-means clustering
            n_clusters = 5
            kmeans1 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels1)
            kmeans2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels2)
            
            # Reshape back to image dimensions
            clusters1 = kmeans1.labels_.reshape(rgb1.shape[:2])
            clusters2 = kmeans2.labels_.reshape(rgb2.shape[:2])
            
            # Analyze cluster statistics to identify debris clusters
            cluster_means1 = []
            cluster_means2 = []
            
            for cluster_id in range(n_clusters):
                # Calculate mean RGB values for each cluster
                mask1 = clusters1 == cluster_id
                mask2 = clusters2 == cluster_id
                
                if np.sum(mask1) > 0:
                    cluster_means1.append(np.mean(rgb1[mask1], axis=0))
                else:
                    cluster_means1.append(np.zeros(3))
                    
                if np.sum(mask2) > 0:
                    cluster_means2.append(np.mean(rgb2[mask2], axis=0))
                else:
                    cluster_means2.append(np.zeros(3))
            
            # Convert to numpy arrays
            cluster_means1 = np.array(cluster_means1)
            cluster_means2 = np.array(cluster_means2)
            
            # Calculate brightness for each cluster
            brightness1 = np.mean(cluster_means1, axis=1)
            brightness2 = np.mean(cluster_means2, axis=1)
            
            # Sort clusters by brightness
            sorted_indices1 = np.argsort(brightness1)
            sorted_indices2 = np.argsort(brightness2)
            
            # Select middle clusters as potential debris
            debris_clusters1 = [sorted_indices1[1], sorted_indices1[2]]
            debris_clusters2 = [sorted_indices2[1], sorted_indices2[2]]
            
            # Create debris masks - ensure they're 2D
            debris_mask1 = np.zeros_like(clusters1, dtype=np.uint8)
            debris_mask2 = np.zeros_like(clusters2, dtype=np.uint8)
            
            for cluster_id in debris_clusters1:
                debris_mask1[clusters1 == cluster_id] = 1
                
            for cluster_id in debris_clusters2:
                debris_mask2[clusters2 == cluster_id] = 1
            
            # 3. Combine clean ice and debris masks - now both are 2D
            combined_glacier1 = np.logical_or(glacier_mask1, debris_mask1).astype(np.uint8)
            combined_glacier2 = np.logical_or(glacier_mask2, debris_mask2).astype(np.uint8)
            
            # 4. Apply morphological operations to clean up masks
            kernel = np.ones((5, 5), np.uint8)
            combined_glacier1 = cv2.morphologyEx(combined_glacier1, cv2.MORPH_CLOSE, kernel)
            combined_glacier2 = cv2.morphologyEx(combined_glacier2, cv2.MORPH_CLOSE, kernel)
            
            # 5. Generate change mask (1 for change, 0 for no change)
            change_mask = np.abs(combined_glacier1 - combined_glacier2).astype(np.uint8)
            
            # 6. Save masks
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_clean_glacier1.npy", glacier_mask1)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_clean_glacier2.npy", glacier_mask2)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_debris_mask1.npy", debris_mask1)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_debris_mask2.npy", debris_mask2)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_combined_glacier1.npy", combined_glacier1)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_combined_glacier2.npy", combined_glacier2)
            np.save(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_change_mask.npy", change_mask)
            
            # 7. Visualize changes for verification
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Display RGB from year 1
            axes[0, 0].imshow(rgb1)
            axes[0, 0].set_title(f"RGB {year1}")
            
            # Display RGB from year 2
            axes[0, 1].imshow(rgb2)
            axes[0, 1].set_title(f"RGB {year2}")
            
            # Display combined glacier mask from year 1
            axes[0, 2].imshow(combined_glacier1, cmap='gray')
            axes[0, 2].set_title(f"Glacier Mask {year1}")
            
            # Display combined glacier mask from year 2
            axes[1, 0].imshow(combined_glacier2, cmap='gray')
            axes[1, 0].set_title(f"Glacier Mask {year2}")
            
            # Display change mask
            axes[1, 1].imshow(change_mask, cmap='hot')
            axes[1, 1].set_title(f"Change {year1}-{year2}")
            
            # Display overlay of change on RGB
            rgb_overlay = rgb2.copy()
            # Add red channel to highlight changes
            rgb_overlay[:, :, 0] = np.where(change_mask > 0, 1.0, rgb_overlay[:, :, 0])
            axes[1, 2].imshow(rgb_overlay)
            axes[1, 2].set_title(f"Change Overlay on {year2}")
            
            plt.tight_layout()
            plt.savefig(f"{data_dir}/change_masks/khumbu_{year1}_{year2}_visualization.png")
            plt.close()
            
            print(f"Generated change detection masks for Khumbu between {year1} and {year2}")
            
        except Exception as e:
            print(f"Error generating change masks for {year1}-{year2}: {str(e)}")
    
    print("Change masks generated successfully")    

def create_khumbu_training_dataset(years, tile_size=256, overlap=128):
    """
    Create training dataset from the downloaded data and generated masks
    
    Parameters:
    - years: list of years
    - tile_size: size of image tiles for training
    - overlap: overlap between tiles
    """
    # Create dataset directories
    dataset_dir = "khumbu_change_dataset"
    os.makedirs(f"{dataset_dir}/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/masks", exist_ok=True)
    
    data_dir = "khumbu_glacier_data"
    image_count = 0
    
    # Process pairs of years
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        # Check if change mask exists
        change_mask_path = f"{data_dir}/change_masks/khumbu_{year1}_{year2}_change_mask.npy"
        if not os.path.exists(change_mask_path):
            print(f"Change mask not found for Khumbu between {year1} and {year2}")
            continue
        
        # Paths to optical, SAR, and masks
        optical1_path = f"{data_dir}/optical/khumbu_{year1}_optical.npy"
        optical2_path = f"{data_dir}/optical/khumbu_{year2}_optical.npy"
        sar1_path = f"{data_dir}/sar/khumbu_{year1}_sar.npy"
        sar2_path = f"{data_dir}/sar/khumbu_{year2}_sar.npy"
        ndsi1_path = f"{data_dir}/ndsi/khumbu_{year1}_ndsi.npy"
        ndsi2_path = f"{data_dir}/ndsi/khumbu_{year2}_ndsi.npy"
        ndwi1_path = f"{data_dir}/ndwi/khumbu_{year1}_ndwi.npy"
        ndwi2_path = f"{data_dir}/ndwi/khumbu_{year2}_ndwi.npy"
        
        # Load data
        optical1 = np.load(optical1_path)
        optical2 = np.load(optical2_path)
        sar1 = np.load(sar1_path)
        sar2 = np.load(sar2_path)
        ndsi1 = np.load(ndsi1_path)
        ndsi2 = np.load(ndsi2_path)
        ndwi1 = np.load(ndwi1_path)
        ndwi2 = np.load(ndwi2_path)
        change_mask = np.load(change_mask_path)
        
        # Prepare feature stack
        # For optical: select RGB and NIR bands (indices 0, 1, 2, 4 from the original data)
        rgb_nir1 = np.stack([optical1[:,:,0], optical1[:,:,1], optical1[:,:,2], optical1[:,:,4]], axis=2)
        rgb_nir2 = np.stack([optical2[:,:,0], optical2[:,:,1], optical2[:,:,2], optical2[:,:,4]], axis=2)
        
        # Combine all features (16 bands total)
        all_features = np.concatenate([
            rgb_nir1, rgb_nir2,  # 8 bands (RGB+NIR from both dates)
            sar1, sar2,         # 4 bands (VV+VH from both dates)
            ndsi1, ndsi2,       # 2 bands (NDSI from both dates)
            ndwi1, ndwi2        # 2 bands (NDWI from both dates)
        ], axis=2)
        
        # Image dimensions
        height, width = all_features.shape[:2]
        
        # Create tiles with overlap
        for y in range(0, height - tile_size + 1, tile_size - overlap):
            for x in range(0, width - tile_size + 1, tile_size - overlap):
                # Extract tile
                tile_features = all_features[y:y+tile_size, x:x+tile_size, :]
                tile_mask = change_mask[y:y+tile_size, x:x+tile_size]
                
                # Skip tiles with little information
                if np.isnan(tile_features).any():
                    continue
                
                # Ensure we include both positive (change) and negative (no change) examples
                # We'll slightly bias toward keeping tiles with changes
                if np.count_nonzero(tile_mask) < 10 and np.random.random() > 0.3:
                    continue  # Skip some no-change tiles (keep 30%)
                
                # Save tile as training example
                np.save(f"{dataset_dir}/images/khumbu_{year1}_{year2}_{x}_{y}.npy", tile_features)
                np.save(f"{dataset_dir}/masks/khumbu_{year1}_{year2}_{x}_{y}.npy", tile_mask)
                
                image_count += 1
    
    print(f"Created {image_count} training examples")
    
    # Perform data augmentation if we don't have enough samples
    if image_count < 1000:
        print("Performing data augmentation to increase dataset size...")
        augment_dataset(dataset_dir)
    
    return dataset_dir

def augment_dataset(dataset_dir):
    """
    Augment the dataset to increase the number of training samples
    Simple augmentations: flip, rotate
    """
    # List original files
    image_files = sorted(os.listdir(f"{dataset_dir}/images"))
    mask_files = sorted(os.listdir(f"{dataset_dir}/masks"))
    
    if len(image_files) == 0:
        print("No images to augment")
        return
    
    augmented_count = 0
    
    # Apply simple augmentations
    for img_file, mask_file in zip(image_files, mask_files):
        try:
            # Load original data
            image = np.load(f"{dataset_dir}/images/{img_file}")
            mask = np.load(f"{dataset_dir}/masks/{mask_file}")
            
            # Get base filename without extension
            base_name = img_file.replace(".npy", "")
            
            # 1. Horizontal flip
            img_flip_h = np.flip(image, axis=1)
            mask_flip_h = np.flip(mask, axis=1)
            np.save(f"{dataset_dir}/images/{base_name}_flip_h.npy", img_flip_h)
            np.save(f"{dataset_dir}/masks/{base_name}_flip_h.npy", mask_flip_h)
            augmented_count += 1
            
            # 2. Vertical flip
            img_flip_v = np.flip(image, axis=0)
            mask_flip_v = np.flip(mask, axis=0)
            np.save(f"{dataset_dir}/images/{base_name}_flip_v.npy", img_flip_v)
            np.save(f"{dataset_dir}/masks/{base_name}_flip_v.npy", mask_flip_v)
            augmented_count += 1
            
            # 3. 90-degree rotation
            img_rot90 = np.rot90(image)
            mask_rot90 = np.rot90(mask)
            np.save(f"{dataset_dir}/images/{base_name}_rot90.npy", img_rot90)
            np.save(f"{dataset_dir}/masks/{base_name}_rot90.npy", mask_rot90)
            augmented_count += 1
            
            # 4. 180-degree rotation
            img_rot180 = np.rot90(image, k=2)
            mask_rot180 = np.rot90(mask, k=2)
            np.save(f"{dataset_dir}/images/{base_name}_rot180.npy", img_rot180)
            np.save(f"{dataset_dir}/masks/{base_name}_rot180.npy", mask_rot180)
            augmented_count += 1
            
            # 5. 270-degree rotation
            img_rot270 = np.rot90(image, k=3)
            mask_rot270 = np.rot90(mask, k=3)
            np.save(f"{dataset_dir}/images/{base_name}_rot270.npy", img_rot270)
            np.save(f"{dataset_dir}/masks/{base_name}_rot270.npy", mask_rot270)
            augmented_count += 1
            
        except Exception as e:
            print(f"Error augmenting {img_file}: {str(e)}")
    
    print(f"Added {augmented_count} augmented samples")

def analyze_khumbu_dataset(dataset_dir):
    """Analyze the created dataset and print statistics"""
    image_files = os.listdir(f"{dataset_dir}/images")
    mask_files = os.listdir(f"{dataset_dir}/masks")
    
    if len(image_files) == 0 or len(mask_files) == 0:
        print("No dataset files found for analysis")
        return
    
    print(f"Total number of images: {len(image_files)}")
    print(f"Total number of masks: {len(mask_files)}")
    
    # Load a sample image to get dimensions and number of features
    sample_image = np.load(f"{dataset_dir}/images/{image_files[0]}")
    sample_mask = np.load(f"{dataset_dir}/masks/{mask_files[0]}")
    
    print(f"Image shape: {sample_image.shape}")
    print(f"Mask shape: {sample_mask.shape}")
    
    # Count positive and negative examples
    positive_count = 0
    negative_count = 0
    
    for mask_file in mask_files:
        mask = np.load(f"{dataset_dir}/masks/{mask_file}")
        if np.any(mask > 0):
            positive_count += 1
        else:
            negative_count += 0
    
    print(f"Positive examples (with change): {positive_count} ({positive_count/len(mask_files)*100:.2f}%)")
    print(f"Negative examples (no change): {negative_count} ({negative_count/len(mask_files)*100:.2f}%)")
    
    # Visualize a few examples
    num_examples = min(5, len(image_files))
    plt.figure(figsize=(15, 4 * num_examples))
    
    for i in range(num_examples):
        image_file = image_files[i]
        mask_file = mask_files[i]
        
        image = np.load(f"{dataset_dir}/images/{image_file}")
        mask = np.load(f"{dataset_dir}/masks/{mask_file}")
        
        # Display RGB from time 1
        plt.subplot(num_examples, 4, i*4 + 1)
        plt.imshow(image[:, :, :3])  # RGB from time 1
        plt.title(f"RGB (t1)")
        plt.axis('off')
        
        # Display RGB from time 2
        plt.subplot(num_examples, 4, i*4 + 2)
        plt.imshow(image[:, :, 4:7])  # RGB from time 2
        plt.title(f"RGB (t2)")
        plt.axis('off')
        
        # Display change mask
        plt.subplot(num_examples, 4, i*4 + 3)
        plt.imshow(mask, cmap='hot')
        plt.title(f"Change Mask")
        plt.axis('off')
        
        # Display overlay
        plt.subplot(num_examples, 4, i*4 + 4)
        rgb_overlay = image[:, :, 4:7].copy()  # RGB from time 2
        rgb_overlay[:, :, 0] = np.where(mask > 0, 1.0, rgb_overlay[:, :, 0])  # Highlight changes in red
        plt.imshow(rgb_overlay)
        plt.title("Change Overlay")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/sample_examples.png")
    plt.close()

if __name__ == "__main__":
    # Define years range for Khumbu Glacier
    years = [2017,2018, 2019, 2020, 2021, 2022,2023]
    
    try:
        use_sentinel_hub = True
        
        # 1. Try to download data from Sentinel Hub first
        if use_sentinel_hub:
            print("Step 1: Downloading multi-temporal data for Khumbu Glacier...")
            try:
                data_dir = download_khumbu_temporal_data(years)
                print(f"Downloaded data saved to: {data_dir}")
            except Exception as e:
                print(f"Sentinel Hub download failed: {str(e)}")
                print("Falling back to synthetic data generation...")
                data_dir = create_synthetic_sentinel_data()
                print(f"Synthetic data created in: {data_dir}")
        else:
            # Use synthetic data directly
            print("Step 1: Creating synthetic Sentinel-1 and Sentinel-2 data...")
            data_dir = create_synthetic_sentinel_data()
            print(f"Synthetic data created in: {data_dir}")
        
        # 2. Generate change detection masks
        print("\nStep 2: Generating change detection masks...")
        generate_khumbu_change_masks(years)
        print("Change masks generated successfully")
        
        # 3. Create training dataset
        print("\nStep 3: Creating training dataset...")
        dataset_dir = create_khumbu_training_dataset(years)
        print(f"Training dataset created at: {dataset_dir}")
        
        # 4. Analyze created dataset
        print("\nStep 4: Analyzing dataset statistics...")
        analyze_khumbu_dataset(dataset_dir)
        print("Dataset analysis complete")
        
        print("\nDataset creation pipeline complete!")
        print(f"The labeled dataset is ready for training at: {dataset_dir}")
        print("Use the 'images' folder for input features and 'masks' folder for labels")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()