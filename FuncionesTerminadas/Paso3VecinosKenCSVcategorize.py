def punto2vecinosk(PuntoDic,kv):
    # -*- coding: utf-8 -*-
    #Esta Funcion recoge los datos de un punto en un dataframe con columnas
    #que tengan Region, type y subType, lat y lng y devuelve los kv vecinos más cercanos y
    #sus caracteristicas en un dataframe

    #Necesita un directorio llamado DatosCategorize que contenga los inmuebles con sus características en csv

    import pandas as pd
    import sklearn.neighbors as sk

    #Cargar el punto como Diccionario
    #PuntoDic={'lat':[41.38465786],'lng':[2.08005229],'Region':["Cataluña"],'type':["house"],'subType':["Vacio"]}
    PuntoDic=pd.DataFrame(PuntoDic)

    CSVdireccion="DataCategorize/"+PuntoDic.iloc[0]['Region']+"_"+PuntoDic.iloc[0]['type']+"_"+PuntoDic.iloc[0]['subType']+".csv"

    Datos=pd.read_csv(CSVdireccion,sep=";",index_col=0)
    #print Datos.columns
    LatLong=Datos.loc[:,['lat','lng']].values
    #print LatLong

    Punto=[PuntoDic.iloc[0]['lat'],PuntoDic.iloc[0]['lng']]

    #print LatLong
    tree =sk.BallTree(LatLong,metric="haversine")#Calcula la distancia Habersine de todos los puntos entre si
    #y los ordena en estructura de arbol
    dist, ind = tree.query(Punto, k=kv)
    print Datos.iloc[ind[0],:]
    return Datos.iloc[ind[0],:]

punto2vecinosk({'lat':[41.38465786],'lng':[2.08005229],'Region':["Cataluña"],'type':["house"],'subType':["Vacio"]},15)

