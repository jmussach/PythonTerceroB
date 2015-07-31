__author__ = 'David'

import json
from shapely.geometry import shape,Point
from shapely.geometry import MultiPolygon,mapping

# load GeoJSON file containing sectors
#with ('data/tercerob_layers.geo.json', 'r') as f:
#    js = json.load(f)
import pygeoj
js = pygeoj.load("spain-communities.geojson")

# construct point based on lat/long returned by geocoder
#point1 = Point(43.360106, -5.896892)
import csv
with open('some.csv', 'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print row
#####
from fiona import collection

schema = { 'geometry': 'Point', 'properties': { 'name': 'str' } }
with collection(
    "some.shp", "w", "ESRI Shapefile", schema) as output:
    with open('some.csv', 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = Point(float(row['lon']), float(row['lat']))
            output.write({
                'properties': {
                    'name': row['name']
                },
                'geometry': mapping(point)
            })
####
# check each polygon to see if it contains the point
for feature in js:
    polygon = shape(feature.geometry)
    if polygon.contains(point1):
        print 'Found containing polygon:', feature



