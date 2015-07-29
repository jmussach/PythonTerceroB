__author__ = 'David'
import json

with open('data/GeolocalizacionRegion.json') as f:
    js = json.load(f)

import Shapely



# construct point based on lat/long returned by geocoder
point = Point("40.185119,-4.034681")

# check each polygon to see if it contains the point
for feature in js['features']:
    polygon = shape(feature['geometry'])
    if polygon.contains(point):
        print 'Found containing polygon:', feature