def AddRegion2LanLongData(LanLogData,GeojsonRegion)

__author__ = 'david'
__author__ = 'david'

from shapely.geometry import shape,Point
import pygeoj
import pandas as pd
import numpy as np

DatosTrain=pd.read_csv("data/DatosEntrenFiltrados50.csv",index_col=0)

js = pygeoj.load("data/spain-communities.geojson")

#Seleccionamos las columnas de longitud y latitud
LongLat=DatosTrain.loc[:,['lng','lat']]
DatosTrain["Region"]=pd.Series(np.nan,index=DatosTrain.index)

for i in range(0,len(LongLat.index)):
    PuntoCoord=LongLat.iloc[i]
    Lng=PuntoCoord.values[0]
    Lat=PuntoCoord.values[1]
    #print Lat," y ",Lng

    point1 = Point(Lng,Lat)

# check each polygon to see if it contains the point
    for feature in js:
        polygon = shape(feature.geometry)
        if polygon.contains(point1):
           # print 'Found containing polygon:', feature.properties['name']
            DatosTrain.iloc[i,(len(DatosTrain.columns)-1)]=feature.properties['name']
            #print DatosTrain.loc[i,['Region']]," y ", feature.properties['name']
    print i

return DatosTrain.to_csv("Prueba4.csv",sep=";",encoding="UTF-8")
