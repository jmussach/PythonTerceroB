__author__ = 'david'
__author__ = 'david'

from shapely.geometry import shape,Point
import pygeoj
import pandas as pd

DatosTrain=pd.read_csv("data/DatosEntrenFiltrados.csv",index_col=0)

#Seleccionamos las columnas de longitud y latitud
LongLat=DatosTrain.loc[:,['lng','lat']]

for i in range(1,len(LongLat.index)):
    PuntoCoord=LongLat.iloc[i]
    print PuntoCoord
    Lat=PuntoCoord.values[0]
    Lng=PuntoCoord.values[1]
    print Lat," y ",Lng
    if i==10:
        break



js = pygeoj.load("data/spain-communities.geojson")

point1 = Point(Lng,Lat)


# check each polygon to see if it contains the point
for feature in js:
    polygon = shape(feature.geometry)
    print polygon.contains(point1)
    if polygon.contains(point1):
        print 'Found containing polygon:', feature.properties['name']

