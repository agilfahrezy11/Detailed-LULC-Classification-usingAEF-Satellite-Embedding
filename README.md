# Implementation of Satellite Embedding Data For Detailed Land Cover Land Use Classification

## Overview
This is the scripts for conducted Land Cover Land Use (LULC) Classification using  [Alpha Earth Foundation Satellite data Embeedding](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/). The Google Satellite Embedding model integrates diverse Earth observation datasets, including:
* Optical Imagery (Landsat-8/9 and Sentinel-2)
* Radar (Sentinel-1 and PALSAR-2)
* Global Ecosystem Dynamics Investigation/GEDI (LIDAR)
* Gravity field (Gravity Recovery and Climate Experiment/GRACE)
* Global elevation models (COPERNICUS GLO-30)
* climate information (ERA-5LAND)
* National Land Cover Database (United States)
* Descriptive Text
<br>
The model produces 64-dimensional embedding vectors at 10 m spatial resolution, providing a detailed and unified representation of Earth’s surface. This dataset streamlines the preprocessing stage for many remote sensing applications, allowing researchers to focus on classification, analysis, and post-processing. Please refer to [Brown et al 2025](https://arxiv.org/abs/2507.22291) for further reading regarding the model creation and dataset.

In this implementation, we apply the embedding data for supervised classification of detailed LULC classes following the Indonesia National Standard Classification scheme, using the Google Earth Engine (GEE) Python API.

## The Data Preparation
In traditional remote sensing workflows, preprocessing involves multiple steps such as dataset selection, cloud masking, spectral index calculation, terrain feature extraction, and multi-source data integration.
Key dataset details:
1. Coverage: Each image covers approximately 163.84 km × 163.84 km
2. Bands: 64 embedding bands (A00 to A63), each representing one axis of the 64D embedding space
3. Temporal resolution: Annual composites 

Below is a helper function (located in gee_utils.py) that retrieves and mosaics the annual embedding images for a given Area of Interest (AOI) and time range.
```python
def get_sat_embedding(aoi, start_year, end_year):
     """
    Retrieves the satellite embedding data
    for the specified time range and area of interest (AOI).

    Parameters:
        aoi (ee.FeatureCollection): 
        start_year (int): 
        end_year (int): 

    Returns:
        ee.Image: A median-composited, clipped image.
    """
     dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

     image = dataset\
        .filterDate(f'{start_year}-01-01', f'{end_year}-01-01') \
         .filterBounds(aoi)
     mosaicked = image.mosaic().clip(aoi)
     return mosaicked
```
We can used this function in a Jupyter Notebook enviroment as shown below:
```python
#example of Area of Interest 
aoi = ee.FeatureCollection('projects/ee-agilakbar/assets/AOI_Garsel_L9')
# Get embeddings for two different periods
image_2024 = get_sat_embedding(aoi, 2024, 2025)
#add visualization parameter
vis_params = {'min': -0.3, 'max': 0.3, 
              'bands': ['A01', 'A16', 'A09']}
```
Here are the result of the visualization of the satellite embedding dataset from the previous code

<p align="center">
  <img src="Detailed-LULC-Classification-usingAEF-Satellite-Embedding\assets\image.png" width="400" alt="Satellite Embedding Visualization">
</p>
