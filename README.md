# Implementation of Google Satellite Embedding for Detailed Land Cover and Land Use Classification in Indonesia
## Overview and Datase Description
This repository demonstrates the implementation of the Google Satellite Embedding dataset for supervised detailed Land Cover and Land Use (LULC) classification. The classification follows the Indonesian National Standard (SNI) Classification Scheme, with training data derived from visual interpretation and field surveys.

This dataset is part of [AlphaEarth Foundation Project](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) by Google DeepMind. It integrate petabytes of earth observation data, as well as ancillary data relating to earth surface dynamics into a single high-dimentional representation. This data comprise of:
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

In this repository, the dataset is used as the primary input for supervised classification . All processing steps — from data retrieval to classification — are automated through helper functions in gee_utils.py.

## Importing The Satellite Embedding Dataset
In traditional remote sensing workflows, preprocessing involves multiple steps such as dataset selection, cloud masking, spectral index calculation, terrain feature extraction, and multi-source data integration.
Key dataset details:
1. Coverage: Each image covers approximately 163.84 km × 163.84 km
2. Bands: 64 embedding bands (A00 to A63), each representing one axis of the 64D embedding space
3. Temporal resolution: Annual composites 

Below is a helper function (located in gee_utils.py) that retrieves and mosaics the annual embedding images for a given Area of Interest (AOI) and time range. Mosaicking the image is crucial since the dataset comprise of various image tiles and your AOI might not be covered in a single time frame. Alternatively, you coud use reducer function to create annual composite of the data.

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
Example Usage
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
  <img src="/assets/image.png" width="600" alt="Satellite Embedding Visualization">
</p>

## Training Data
For this project, I used 700 labeled polygons (≈13,000 pixels) prepared through visual interpretation and field work during my master’s thesis.

To prepare the data for classification and hyperparameter optimization, I created a helper function that:

* Automatically extracts pixel values from the embedding image based on predefined Regions of Interest (ROI)
* Randomly splits the data into training and testing sets

This ensures reproducible sampling and flexible control over the split ratio.
The function is available in gee_utils.py.
```python
def extract_pixel_value(image, roi, class_property, scale=10, split_ratio = 0.5, tile_scale = 16):
     """
     extract the pixels value for supervised classification based on pre defined regions of interest (ROI) and
     partitioned the data into training and testing
     Parameters:
        image (ee.Image)
        Region of interest/ROI (feature collection)
        class_property/class_label
        scale(int), based on spatial resolution of the data
        split_ratio (float)
        tile_scale
     Returns:
        tuple: (training_samples, testing_samples)
     """
     #create a random column
     roi_random = roi.randomColumn()
     #partioned the original training data
     training = roi_random.filter(ee.Filter.lt('random', split_ratio))
     testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
     #extract the pixel values
     training_pixels = image.sampleRegions(
                        collection=training,
                        properties = [class_property],
                        scale = scale,
                        tileScale = tile_scale 
     )
     testing_pixels = image.sampleRegions(
                        collection=testing,
                        properties = [class_property],
                        scale = scale,
                        tileScale = tile_scale 
     )
     return training_pixels, testing_pixels
```
Example usage:
```python
# defining the training data
labeled_roi = ee.FeatureCollection('projects/ee-agilakbar/assets/LULC_Training_data_Garsel_project')
# partitioned the ROI into training and testing data, then extract the pixel values (sample regions)
train_samples, test_samples = extract_pixel_value(
    image=image_2024,
    roi=labeled_roi,
    class_property='LUCID',
    scale=10,
    split_ratio=0.7,
    tile_scale=16
)
```
## Hyperparameter optimization
One of the most crucial part of machine learning workflow is finding the most optimal paramaters for any given task. The optimal paramater combination ensure that the model captured the pattern of the data, ensuring robust and scalabel learning. The Google Earth Engine used the Statistical Machine Intelligence Learning (SMILE) which is a javascript library for conducting various machine learning related task. 


