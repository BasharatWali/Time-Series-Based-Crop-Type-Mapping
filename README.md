This project focuses on mapping different crop types over time using monthly Sentinel-2 imagery for Ontario, Canada. By integrating temporal information from multiple months (May through October) over several years (2018–2021), we aim to capture the distinctive phenological patterns that each crop exhibits throughout the growing season. Our approach uses a custom-designed U-Net architecture that incorporates both spatial and temporal dimensions to accurately segment and classify 6 different crop types. The primary outcome is a set of georeferenced maps showing the distribution of these crops, which can support agricultural monitoring, resource planning, and policy-making efforts.

Data and Study Area
    Location: Ontario, Canada
    Satellite Data: Sentinel-2 monthly composites
    Time Frames:
        Training: 3 years of data (2018–2020)
        Testing: 1 year of data (2021)
    Months Considered: May through October (6 distinct time slices in each growing season)
    Crops: 6 major crop types.

Data Acquisition and Preprocessing
1. Sentinel Hub: Satellite imagery was sourced and preprocessed to minimize effects of cloud cover, atmospheric distortions, and varying illumination conditions.
2. Temporal Stacking: Each pixel’s reflectance values for the 6 monthly time slices were stacked, creating a 4D dataset (time×height×width×bands)(time×height×width×bands).
3. Labeling: Ground-truth data for crop types were collected from official agricultural databases and field surveys. Masks delineating different crop types were generated and aligned with the satellite data.
4. Patch Creation: To train the U-Net, the large satellite scenes were divided into smaller patches (e.g., 64×64 pixels), with each patch retaining the 6 monthly time slices.
