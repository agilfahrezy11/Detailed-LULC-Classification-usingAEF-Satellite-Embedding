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
  <img src="/assets/embed_vis.png" width="600" alt="Satellite Embedding Visualization">
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
One of the most crucial part of machine learning workflow is finding the most optimal paramaters for any given task. Choosing the right parameter values ensures that the model captures meaningful patterns in the data, leading to robust and scalable predictions. The Google Earth Engine machine learning algorithms are implemented through the Statistical Machine Intelligence and Learning Engine (SMILE) — a JavaScript-based library for classification, regression, and clustering tasks. Unlike libraries such as Scikit-learn, SMILE does not provide built-in utilities for automated hyperparameter search (e.g., GridSearchCV). As a result, hyperparameter optimization in GEE requires a manual search approach, which consist of fallowing steps:

1. Define a set of possible parameter combinations
2. Train and evaluate a model for each combination
3. Compare performance metrics 
4. Select the combination that yields the best results

This repository implements a helper function to streamline the process, allowing you to systematically test different parameter configurations and identify the most effective setup for your LULC classification task.
```python
def rf_tuning(train, test, band_names, class_property, n_tree_list, var_split_list, min_leaf_pop_list):
         for n_tree in n_tree_list:
          for var_split in var_split_list:
               for min_leaf_pop in min_leaf_pop_list:
                  try:
                    #initialize the random forest classifer
                    clf = ee.Classifier.smileRandomForest(
                         numberOfTrees=n_tree,
                         variablesPerSplit=var_split,
                         minLeafPopulation = min_leaf_pop,
                         seed=0
                    ).train(
                        features=train,
                        classProperty=class_property,
                        inputProperties=band_names
                    )
                     #Used partitioned test data, to evaluate the trained model
                    classified_test = test.classify(clf)
                    #test using error matrix
                    error_matrix = classified_test.errorMatrix(class_property, 'classification')
                    #append the result of the test
                    accuracy = error_matrix.accuracy().getInfo()
                    result.append({
                        'numberOfTrees': n_tree,
                        'variablesPerSplit': var_split,
                        'MinimumleafPopulation0':min_leaf_pop,
                        'accuracy': accuracy
                    })
                    #print the message if error occur
                  except Exception as e:
                    print(f"Failed for Trees={n_tree}, Variable Split={var_split}, mininum leaf population = {min_leaf_pop}")
                    print(e)
               return result
```
Example Usage:

```python
#Parameter space
num_trees = [100, 200, 300]
var_split = [1,2,3,8,11,15]
min_leaf_pop = [1,2,3, 5, 9, 11]
#apply the function
results_rf = rf_tuning(
    train=train_samples, #Training data, with pixel value
    test=test_samples, #testingd data with pixel value
    band_names=image_2024.bandNames(), #band names 
    class_property='LUCID', #class labels
    n_tree_list=num_trees, #tree parameter space
    var_split_list=var_split, #variable split space
    min_leaf_pop_list=min_leaf_pop #minimum leaf pop space
)
```

In this example, three hyperparameters were tested: number of trees, number of variables selected at each split, and minimum leaf population. One important caveat is that increasing the number of parameters or the range of values tested will significantly increase computation time. To inspect the best parameters combination we could use the following code:
```Python
#convert the result into panda data frame
df_rf = pd.DataFrame(results_rf)
df_rf = df_rf.sort_values(by='accuracy', ascending=False)
print(df_rf.head())
#get the best parameters and accuracy
best_result = df_rf.iloc[0]
print("Best Hyperparameters:")
print(f"- Accuracy: {best_result['accuracy']:.3f}")
print(f"- numberOfTrees: {best_result['numberOfTrees']}")
print(f"- variablesPerSplit: {best_result['variablesPerSplit']}")
print(f"- Minimum number of leaf population: {best_result ['MinimumleafPopulation0']}")
```

Here are the result of the hyperparameter optimization:

<p align="center">
  <img src="/assets/image.png" width="600" alt="Satellite Embedding Visualization">
</p>

As we can see, the combination of 100 tree, 1 variable split, and 2 minimum popluation, resulted in an accuracy 89%, which can be considered substantial. 

## Applied the best parameters and classified the imagery

Next, we applied these parameters for classifying the whole imagery using the following code:

```python
#get the best parameters
best_ntree = int(best_result['numberOfTrees'])
best_vsplit = int(best_result['variablesPerSplit'])
best_min_leaf = int(best_result['MinimumleafPopulation0'])
#apply the best parameters, and trained the model
final_classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=best_ntree,
    variablesPerSplit=best_vsplit,
    minLeafPopulation=best_min_leaf
).train(
    features=train_samples,
    classProperty='LUCID',
    inputProperties=image_2024.bandNames()
)
#apply the trained model into the whole image
lulc_map = image_2024.classify(final_classifier)
```
To visualize the classified result, we can use the following code:

```python
legend_21 = [
    'Settlement Build-up', 'Non-settlement Build-up', 'Volcanic Rock and Sand', 'Lowland Forest',
    'Upland Forest', 'Mangrove Forest', 'Marine Sand', 'Herbs and Grassland', 'Production Forest',
    'Aquaculture', 'Dryland Agriculture', 'Natural Fallow Land', 'Man-made Fallow Land', 'Rubber and Other Hardwood Plantation',
    'Ocean Waters','Oil Palm Plantatation','Tea Plantation', 'Bushes and Shrubs', 'Rivers', 'Wetland Agriculture',
    'Natural/Semi Natural Freshwaterbody'
]

legend_colors = [
    '#FF0000', '#e97421', '#7b3531', '#2e8b57', '#228b22', '#7fffd4', '#fdd9b5',
    '#c1e1c1', '#9acd32', '#ccccff', '#f5ff00', '#b5a642', '#cc8899', '#32cd32',
    '#1f52dc', '#808000', '#98d97d', '#008080', '#87ceeb', '#afb325', '#0f3c5e'
]
Map = geemap.Map()

# Visualization
vis_params = {
    'min': 0,
    'max': len(legend_colors) - 1,
    'palette': legend_colors
}
Map.centerObject(aoi, 10)
Map.addLayer(lulc_map, vis_params, 'Classified Map')
# Add legend
Map.add_legend(title='Land Cover Legend', labels=legend_21, colors=legend_colors)
Map
```
## Exporting the result
If you're not satisfied with the visual representation of the land cover land use class, you can export the result to google drive and download the tiff file. To do this, you can use the following code:
```python
export_task = ee.batch.Export.image.toDrive(
    image=lulc_map,
    description='Update_RF_LC_sat_embed',
    folder='Earth Engine',
    fileNamePrefix='Update_RF_LC_sat_embed',
    scale=10,
    region=image_2024.geometry(),  # or aoi.geometry()
    maxPixels=1e13
)
#Monitored the progress
export_task.start()
import time

while export_task.active():
    print('Exporting... (status: {})'.format(export_task.status()['state']))
    time.sleep(10)

print('Export complete (status: {})'.format(export_task.status()['state']))
```

