__author__ = 'david'

import pandas as pd
#Datos=pd.read_csv("DataInput/TrainWithRegion1000.csv",sep=";",index_col=0)
Datos=pd.read_csv("DataInput/DatosConComunidad.csv",sep=";",index_col=0)
Datos=pd.DataFrame(Datos)
#DatosCat=pd.Series(Datos['type'].astype("string"))
DatosType = Datos["type"]

LevelsType=set(DatosType)

DatosRegion = Datos["Region"]

LevelsRegion =list(set(DatosRegion))
for province in LevelsRegion:
    for typebuild in LevelsType:
        DatosFiltrados= Datos[(Datos.type==typebuild)&(Datos.Region==province)]
        CsvFileName="DataOutput/"+str(province)+str(typebuild)+".csv"
        #CsvFileName="outputData/"+'%s'+'.csv' % str(province)#str(typebuild)
        DatosFiltrados.to_csv(CsvFileName,sep=";",encoding="UTF-8")