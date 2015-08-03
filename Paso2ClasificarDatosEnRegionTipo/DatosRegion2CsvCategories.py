def DatosRegion2CsvCategories(DatosRegion,NomColumFilter ):
    #Es una función que a partir de un csv con datos con regiones "DatosRegion" los filtra por regiones,
#y tipos y los guarda en multiples csv según los nombres de columna escogidos

import pandas as pd


Datos=pd.read_csv(DatosRegion,sep=";",index_col=0)''
Datos=pd.DataFrame(Datos)

for Columnas in NomClomunFilter:

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

DatosRegion2CsvCategories("DataInput/TrainWithRegion1000.csv", ["Region","type"])