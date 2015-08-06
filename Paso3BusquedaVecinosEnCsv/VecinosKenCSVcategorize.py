def punto2vecinosk(PuntoDic,kv):
    # -*- coding: utf-8 -*-
    #Esta Funcion recoge los datos de un punto en un dataframe con columnas
    #que tengan Region, type y subType, lat y lng y devuelve los kv vecinos más cercanos y
    #sus caracteristicas en un dataframe

    #Necesita un directorio llamado DatosCategorize que contenga los inmuebles con sus características en csv

    import pandas as pd
    import sklearn.neighbors as sk

    CSVdireccion="DataCategorize/"+PuntoDic['Region']+"_"+PuntoDic['type']+"_"+PuntoDic['subType']+".csv"

    Datos=pd.read_csv(CSVdireccion,sep=";",index_col=0,encoding="UTF-8")

    LatLong=Datos.loc[:,['lat','lng']].values

    Punto=[PuntoDic['lat'],PuntoDic['lng']]

    tree =sk.BallTree(LatLong,metric="haversine")#Calcula la distancia Habersine de todos los puntos entre si
    #y los ordena en estructura de arbol
    dist, ind = tree.query(Punto, k=kv)
    return Datos.iloc[ind[0]]


import pandas as pd
Datos=pd.read_csv("DataInput/DatosAValorarConRegion.csv",sep=";",index_col=0,encoding="UTF-8")

print punto2vecinosk(Datos.iloc[2],6)

#punto2vecinosk({'lat':[41.38465786],'lng':[2.08005229],'Region':["Cataluña"],'type':["house"],'subType':["Vacio"]},15)

