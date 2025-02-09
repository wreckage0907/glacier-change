import os
import json
import base64
import requests
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GlacierDatasetBuilder:
    def __init__(self, data_dir='glacier_data'):
        """
        Initialize the dataset builder
        data_dir: Directory to store the dataset
        """
        self.data_dir = data_dir
        self.create_directory_structure()
        
        # Get credentials from environment variables
        self.cdse_user = os.getenv('COPERNICUS_USERNAME')
        self.cdse_password = os.getenv('COPERNICUS_PASSWORD')
        
        if not self.cdse_user or not self.cdse_password:
            raise ValueError(
                "Missing credentials! Please set COPERNICUS_USERNAME and "
                "COPERNICUS_PASSWORD environment variables."
            )
        
        print(f"Initialized with username: {self.cdse_user}")

    def create_directory_structure(self):
        """Create necessary directories for dataset organization"""
        directories = [
            self.data_dir,
            f'{self.data_dir}/raw_images',
            f'{self.data_dir}/processed_images',
            f'{self.data_dir}/metadata',
            f'{self.data_dir}/labels'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created/verified directory: {directory}")

    def download_glacier_data(self, glacier, time_range):
        """
        Download Sentinel-2 imagery with more lenient search criteria
        """
        coords = glacier['coordinates']
        auth = (self.cdse_user, self.cdse_password)
        
        start_date = time_range['start_date'].strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = time_range['end_date'].strftime("%Y-%m-%dT%H:%M:%SZ")
        
        base_url = "https://apihub.copernicus.eu/apihub/search"
        
        # Construct OpenSearch query parameters
        query_params = {
            'q': f'platformname:Sentinel-2 AND cloudcoverpercentage:[0 TO 50] AND beginPosition:[{start_date} TO {end_date}]',
            'format': 'json',
            'rows': 20,
            'bbox': f"{coords['min_lon']},{coords['min_lat']},{coords['max_lon']},{coords['max_lat']}"
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Basic {base64.b64encode(f"{self.cdse_user}:{self.cdse_password}".encode()).decode()}'
        }
        
        print(f"Searching for images with parameters: {json.dumps(query_params, indent=2)}")
        
        try:
            response = requests.get(
                base_url, 
                params=query_params,
                auth=auth,
                headers=headers,
                verify=True  # For SSL verification
            )
            
            # Print the actual URL being queried for debugging
            print(f"Request URL: {response.url}")
            
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                print(f"Response content: {response.text}")
                return []
                
            search_results = response.json()
            if 'entry' not in search_results:
                print("No entries found in search results")
                print(f"Full response: {json.dumps(search_results, indent=2)}")
                return []
                
            products = search_results['entry']
            print(f"Found {len(products)} products")
            
            # Download products
            downloaded_products = []
            for product in products:
                try:
                    product_id = product['title']
                    download_url = product.get('link')[0].get('href')
                    
                    # Get asset information
                    # Get download URL from the product links
                    download_url = None
                    for link in product.get('link', []):
                        if link.get('rel') == 'alternative':
                            download_url = link.get('href')
                            break
                            
                    if not download_url:
                        print(f"No download URL found for {product_id}")
                        continue
                    
                    output_file = f"{self.data_dir}/raw_images/{product_id}.zip"
                    if not os.path.exists(output_file):
                        print(f"Downloading {product_id}...")
                        download_response = requests.get(download_url, auth=auth, stream=True)
                        download_response.raise_for_status()
                        
                        with open(output_file, 'wb') as f:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Downloaded {product_id}")
                    else:
                        print(f"Using cached file for {product_id}")
                    
                    downloaded_products.append(product)
                        
                except Exception as e:
                    print(f"Error processing product {product_id if 'product_id' in locals() else 'unknown'}: {str(e)}")
                    continue
            
            return downloaded_products
            
        except Exception as e:
            print(f"Error making request: {str(e)}")
            return []

    def process_raw_images(self, image_path, output_size=(224, 224)):
        """
        Process raw satellite images for CNN input
        
        image_path: Path to raw satellite image
        output_size: Desired size of processed images
        """
        print(f"Processing image: {image_path}")
        try:
            with rasterio.open(image_path) as src:
                # Read the green, red, and NIR bands
                image = np.dstack([src.read(3), src.read(4), src.read(8)])
                
                # Normalize values to 0-255 range
                for i in range(3):
                    band = image[:,:,i]
                    min_val = np.percentile(band, 2)
                    max_val = np.percentile(band, 98)
                    image[:,:,i] = np.clip(255 * (band - min_val) / (max_val - min_val), 0, 255)

                # Resize image
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize(output_size, Image.Resampling.LANCZOS)
                
                print(f"Successfully processed image to size {output_size}")
                return np.array(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise

    def calculate_glacier_change(self, initial_image, final_image):
        """
        Calculate glacier change between two timestamps
        Returns: 0 (stable), 1 (retreating), or 2 (advancing)
        """
        try:
            # Calculate NDSI (Normalized Difference Snow Index)
            def calculate_ndsi(image):
                green = image[:,:,0].astype(float)
                swir = image[:,:,2].astype(float)  # Using NIR band as approximate SWIR
                ndsi = (green - swir) / (green + swir + 1e-8)
                return ndsi
            
            initial_ndsi = calculate_ndsi(initial_image)
            final_ndsi = calculate_ndsi(final_image)
            
            # Calculate change metrics
            initial_coverage = np.sum(initial_ndsi > 0.4)
            final_coverage = np.sum(final_ndsi > 0.4)
            
            if initial_coverage == 0:
                print("Warning: No glacier detected in initial image")
                return 0
                
            change_ratio = final_coverage / initial_coverage
            
            # Classify change
            if 0.95 < change_ratio < 1.05:  # 5% threshold
                return 0  # Stable
            elif change_ratio <= 0.95:
                return 1  # Retreating
            else:
                return 2  # Advancing
                
        except Exception as e:
            print(f"Error calculating glacier change: {str(e)}")
            raise

    def build_dataset(self, glacier_list, time_period=730):  # Default to 2 years
        """
        Build complete dataset for multiple glaciers
        
        glacier_list: List of dictionaries containing glacier coordinates and metadata
        time_period: Number of days between initial and final images for change detection
        """
        dataset = []
        labels = []
        
        for glacier in glacier_list:
            print(f"\nProcessing glacier: {glacier['name']}")
            
            # Download data for two time periods
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period)
            
            time_ranges = {
                'initial': {
                    'start_date': start_date - timedelta(days=60),  # Increased window
                    'end_date': start_date + timedelta(days=60)
                },
                'final': {
                    'start_date': end_date - timedelta(days=60),
                    'end_date': end_date
                }
            }
            
            initial_image = None
            final_image = None
            
            for period, date_range in time_ranges.items():
                print(f"\nDownloading {period} period data...")
                products = self.download_glacier_data(glacier, date_range)
                
                if not products:
                    print(f"No products found for {period} period")
                    continue
                    
                # Process each product
                for product in products:
                    try:
                        file_path = f"{self.data_dir}/raw_images/{product['id']}.zip"
                        processed_image = self.process_raw_images(file_path)
                        
                        if period == 'initial':
                            initial_image = processed_image
                            print("Initial image processed successfully")
                        else:
                            final_image = processed_image
                            print("Final image processed successfully")
                            
                        if initial_image is not None and final_image is not None:
                            # Calculate change and add to dataset
                            change_label = self.calculate_glacier_change(
                                initial_image, final_image
                            )
                            
                            dataset.append(final_image)
                            labels.append(int(change_label))  # Convert to integer
                            print(f"Added pair to dataset with label {change_label}")
                            
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                        continue
        
        # Convert to numpy arrays
        X = np.array(dataset)
        y = np.array(labels, dtype=np.int64)  # Explicitly set dtype to int64
        
        print(f"\nFinal dataset size: {len(dataset)} samples")
        
        # Save dataset
        if len(dataset) > 0:
            np.save(f'{self.data_dir}/processed_images/X.npy', X)
            np.save(f'{self.data_dir}/labels/y.npy', y)
            print("Dataset saved successfully")
        else:
            print("Warning: No data was collected!")
        
        return X, y

    def add_metadata(self, glacier_info):
        """
        Add metadata for each glacier in the dataset
        
        glacier_info: Dictionary containing glacier metadata
        """
        try:
            metadata_df = pd.DataFrame(glacier_info)
            metadata_df.to_csv(f'{self.data_dir}/metadata/glacier_metadata.csv', index=False)
            print("Metadata saved successfully")
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Example glacier coordinates (Aletsch Glacier)
    example_glaciers = [
        {
            'name': 'Aletsch Glacier',
            'coordinates': {
                'min_lon': 7.95,  # Slightly adjusted coordinates
                'max_lon': 8.55,
                'min_lat': 46.25,
                'max_lat': 46.55
            },
            'country': 'Switzerland',
            'elevation': 3500
        }
    ]
    
    try:
        # Initialize dataset builder
        builder = GlacierDatasetBuilder()
        
        # Build dataset with 2-year time period
        X, y = builder.build_dataset(example_glaciers)
        
        # Add metadata
        builder.add_metadata(example_glaciers)
        
        print("\nFinal Results:")
        print(f"Dataset shape: {X.shape}")
        if len(y) > 0:
            print(f"Number of samples per class: {np.bincount(y)}")
            print("Classes: 0=Stable, 1=Retreating, 2=Advancing")
        else:
            print("No samples collected")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")