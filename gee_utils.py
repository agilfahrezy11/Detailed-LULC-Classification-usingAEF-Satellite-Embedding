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