# Implementation of Satellite Embedding Data For Detailed Land Cover Land Use Classification

## Overview
This is the scripts for conducted Land Cover Land Use (LULC) Classification using  [Alpha Earth Foundation Satellite data Embeedding](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/). This new model integrate various earth observation data, ranging from optical, radar, LIDAR, 3D measurements of surface properties, global elevation models, climate information, gravity fields, and descriptive text. The model resulted in 64 embedding vectors at 10-m spatial resolution, providing finer view of the earth surface. This model revolutionize the pre-processing stages of many remote sensing application, and as a result, user's can focused on the processing and post stages of their study. This repository provide implementation of supervised classification for mapping detailed Land Cover Land Use class, based on Indonesia National Standard Classification scheme. 
## The Data Preparation
Usually, the data preparation of remote sensing imageries comprise of data selection, filtering, covariates gathering, etc. With Emebedding data, this process is simplified into several line of code. The collection is composed of images covering approximately 163,840 meters by 163,840 meters, and each image has 64 bands (A00, A01, …, A63), one for each axis of the 64D embedding space. Below code shows the function for filtering the data, in which annual data is used for the classification. Since the data in the area of interest (AOI) comprise of several embedding, a mosaic function is implement, ensuring a single image covering the whole AOI.
```python
def get_sat_embedding(aoi, start_year, end_year):
     dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

     image = dataset\
        .filterDate(f'{start_year}-01-01', f'{end_year}-01-01') \
         .filterBounds(aoi)
     mosaicked = image.mosaic().clip(aoi)
     return mosaicked
```
