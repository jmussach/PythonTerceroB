__author__ = 'david'
__author__ = 'david'

from shapely.geometry import shape,Point
import pygeoj
import pandas as pd
import numpy as np


js = pygeoj.load("data/spain-communities.geojson")


point1 = Point(-2.7213783,43.4210531)

# check each polygon to see if it contains the point
for feature in js:
        polygon = shape(feature.geometry)
        if polygon.contains(point1):
            print feature.properties['name']


