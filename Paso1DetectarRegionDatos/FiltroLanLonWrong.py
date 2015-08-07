def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

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
    indWrong=[]
    print len( LongLat)
    for i in range(len( LongLat)):
        Lng=LongLat.iloc[i,0]
        Lat=LongLat.iloc[i,1]
        if (isFloat(Lng)) : Lng=float(Lng)
        if (isFloat(Lat)) : Lat=float(Lat)
        if (((isinstance(Lng,float)==False) | ((Lng>1000) | (Lng<-1000))) | ((isinstance(Lat,float)==False) | ((Lat>1000) | (Lat<-1000))) ):
            indWrong.append(i)
    else:
            indRight.append(i)
    DatosAValorarRight=DatosAValorar.iloc[indRight]
    DatosAValorarWrong=DatosAValorar.iloc[indWrong]
    NombreBase=DatosSinFiltrar.split('.')
    print NombreBase[0]+'Right'+'.csv'
    DatosAValorarRight.to_csv(NombreBase[0]+'Right'+'.csv',sep=";",encoding="UTF-8")
    DatosAValorarWrong.to_csv(NombreBase[0]+'Wrong'+'.csv',sep=";",encoding="UTF-8")


FiltroLatLonWrong("OutputData/DatosAValorar.csv")
