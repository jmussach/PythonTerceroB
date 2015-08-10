def AddRegion2LanLongData(LanLogData,GeojsonRegion,CsvFileName):

    from shapely.geometry import shape,Point
    import pygeoj
    import pandas as pd
    import numpy as np

    DatosTrain=pd.read_csv(LanLogData,index_col=0)

    js = pygeoj.load(GeojsonRegion)

    #Seleccionamos las columnas de longitud y latitud
    LongLat=DatosTrain.loc[:,['lng','lat']]
    DatosTrain["Region"]=pd.Series(np.nan,index=DatosTrain.index)

    for i in range(0,len(LongLat.index)):
        PuntoCoord=LongLat.iloc[i]
        Lng=PuntoCoord.values[0]
        Lat=PuntoCoord.values[1]


        point1 = Point(Lng,Lat)

    # check each polygon to see if it contains the point
        for feature in js:
            polygon = shape(feature.geometry)
            if polygon.contains(point1):
                      DatosTrain.iloc[i,(len(DatosTrain.columns)-1)]=feature.properties['name']


    return DatosTrain.to_csv(CsvFileName,sep=";",encoding="UTF-8")

AddRegion2LanLongData("data/DatosEntrenFiltrados.csv","data/spain-communities.geojson","outputData/DatosEntrenConRegion.csv")
