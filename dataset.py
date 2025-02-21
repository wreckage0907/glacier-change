from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    bbox_to_dimensions,
)
from config import SH_CLIENT_ID, SH_CLIENT_SECRET 
from datetime import datetime, timedelta
import numpy as np
import os

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
    
    try:
        # Test the configuration
        from sentinelhub import DataCollection
        collection = DataCollection.SENTINEL2_L2A
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        raise
    
    return config
def get_khumbu_bbox():
    """Define bounding box for Khumbu Glacier"""
    return BBox(
        bbox=[86.78, 27.95, 86.85, 28.00],
        crs=CRS.WGS84
    )

def get_sentinel2_request(config, bbox, time_interval):
    """Create Sentinel-2 data request with proper scaling"""
    return SentinelHubRequest(
        data_folder="khumbu_glacier_data",
        evalscript="""
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B08", "B11", "B12"],
                        units: "REFLECTANCE"
                    }],
                    output: {
                        bands: 6,
                        sampleType: "FLOAT32"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B02, sample.B03, sample.B04, 
                        sample.B08, sample.B11, sample.B12];
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
        config=config
    )

def get_sentinel1_request(config, bbox, time_interval):
    """Create Sentinel-1 data request"""
    return SentinelHubRequest(
        data_folder="khumbu_glacier_data",
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
        config=config
    )

def download_seasonal_data(year=2024):
    """Download seasonal data for both Sentinel-1 and Sentinel-2"""
    config = setup_sentinel_hub()
    bbox = get_khumbu_bbox()
    
    # Define seasons
    seasons = [
        ('winter', f'{year}-01-01', f'{year}-02-28'),
        ('spring', f'{year}-03-01', f'{year}-05-31'),
        ('summer', f'{year}-06-01', f'{year}-08-31'),
        ('autumn', f'{year}-09-01', f'{year}-11-30')
    ]
    
    # Create output directory
    os.makedirs('khumbu_glacier_data', exist_ok=True)
    
    for season, start_date, end_date in seasons:
        time_interval = (start_date, end_date)
        
        # Download Sentinel-2 data
        s2_request = get_sentinel2_request(config, bbox, time_interval)
        s2_data = s2_request.get_data()
        
        # Download Sentinel-1 data
        s1_request = get_sentinel1_request(config, bbox, time_interval)
        s1_data = s1_request.get_data()
        
        # Save data
        np.save(f'khumbu_glacier_data/s2_{season}_{year}.npy', s2_data[0])
        np.save(f'khumbu_glacier_data/s1_{season}_{year}.npy', s1_data[0])

if __name__ == "__main__":
    try:
        # Test authentication first
        config = setup_sentinel_hub()
        print("Starting data download...")
        download_seasonal_data()
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please verify your Sentinel Hub credentials and subscription status.")