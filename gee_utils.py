import ee

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
         .filterBounds(aoi)\
         .median()\
        .clip(aoi)
     return image

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

def rf_tuning(train, test, band_names, class_property, n_tree_list, var_split_list, min_leaf_pop_list):
     """
     Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier.
     Two main parameters were tested, namely Number of trees (n_tree), and number of variable selected at split (var_split)
     Additional parameters for the function:
         train: Training pixels
         test: Testing pixels
         band_names: band names of remote sensing imagery
         class_property: distinct labels in the training and testing data
     """
     result = [] #initialize empty dictionary for storing parameters and accuracy score
     #manually test the classifiers, while looping through the parameters set
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

