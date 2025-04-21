# glacier-change
A project for detecting glacier changes in the Khumbu Glacier using semantic segmentation with U-Net architecture enhanced by attention mechanisms. This approach enables precise monitoring of glacial retreat and transformation over time through satellite imagery analysis.

The implementation uses Sentinel Hub data to track changes in one of the Himalaya's most well-known glaciers, providing insights into climate change impacts on critical water resources.
## Setup Instructions

1. **Set up Sentinel Hub access**:
    - Visit https://www.sentinel-hub.com/
    - Log in to your account
    - Go to User Settings → Account settings
    - Navigate to "Apps" tab
    - Create a new OAuth client
    - You'll receive:
      - Client ID
      - Client Secret
2.  **Create config.py file**:
    ```bash
    SH_CLIENT_ID = "your-client-id"
    SH_CLIENT_SECRET = "your-client-secret"
    ```

3. **Prepare the dataset**:
    ```
    python3 build_dataset.py
    ```

4. **Train the model**:
    ```
    python3 train.py
    ```
