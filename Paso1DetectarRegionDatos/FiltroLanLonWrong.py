def FiltroLatLonWrong(DatosSinFiltrar):
    #Esta funcion recibe con parametro un csv con columnas de lat y lng y filtra aquellas
#filas que no tienen la estrutura esperada

    from shapely.geometry import shape,Point
    import pygeoj
    import pandas as pd
    import numpy as np

    DatosAValorar=pd.read_csv(DatosSinFiltrar,sep=";",index_col=0,error_bad_lines=False)
    LongLat=DatosAValorar.loc[:,['lng','lat']]
    indRight=[]
    for i in range(len( LongLat)):
        Lng=LongLat.iloc[i,0]
        Lat=LongLat.iloc[i,1]
        if (isinstance(Lng,float) | (Lng>1000) | (Lng<1000)) :
            indRight.append(i)
        if (isinstance(Lng,float) | (Lat>1000) | (Lat<1000)) :
            indRight.append(i)
    #indRightUniq=set(indRight)

    #print indRightUniq


FiltroLatLonWrong("data/DatosAValorar.csv")
