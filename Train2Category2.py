__author__ = 'david'
import json
from shapely.geometry import shape,Point
from shapely.geometry import MultiPolygon,mapping

# load GeoJSON file containing sectors
#with ('data/tercerob_layers.geo.json', 'r') as f:
#    js = json.load(f)
import pygeoj
js = pygeoj.load("spain-communities.geojson")

point1 = Point(40.1851186, -4.034681)

def point_in_poly(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

#point_in_poly(40.445686, -3.734746,js)

# check each polygon to see if it contains the point
for feature in js:
    polygon = shape(feature.geometry)
    #print point_in_poly(40.445686,-3.734746,feature.geometry)
    #print polygon
    print feature.geometry
    if polygon.contains(point1):
        print 'Found containing polygon:', feature



