__author__ = 'david'
import pandas as pd
import numpy as np
    #Cargamos los datos
Datos=pd.read_csv("DataInput/DatosConComunidad.csv",sep=";",index_col=0)

#print Datos.head(20)

Datos=pd.DataFrame(Datos)
Datos=Datos.fillna("Vacio")
#Datos.fillna("")
    #Seleccion datos region y sus niveles
DatosRegion = Datos["Region"]
LevelsRegion =set(DatosRegion)

#Seleccion datos type y sus niveles
DatosType = Datos["type"]
LevelsType=set(DatosType)

#Seleccion datos subType y sus niveles
DatossubType = Datos["subType"]
LevelssubType=set(DatossubType)


#print LevelssubType
#print Datos.columns
#A=[Datos.loc[1,['subtype']] ]
#B=np.empty(len(Datos.subType)
#A.fill(B)
#print len(A)
print len(Datos.subType)
print len(Datos[Datos.subType=="Vacio"])