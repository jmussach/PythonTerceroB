def DatosRegion2CsvCategories(DatosRegion):
    #Es una funcion que a partir de un csv con datos con regiones "DatosRegion" los filtra por regiones,
    #tipos y subtipos los guarda en multiples csv segun la Categoria Region-Tipo-Subtipo

    import pandas as pd

    #Cargamos los datos
    Datos=pd.read_csv(DatosRegion,sep=";",index_col=0)
    Datos=pd.DataFrame(Datos)

    #Seleccion datos region y sus niveles
    DatosRegion = Datos["Region"]
    LevelsRegion =set(DatosRegion)

    #Seleccion datos type y sus niveles
    DatosType = Datos["type"]
    LevelsType=set(DatosType)

    #Seleccion datos subType y sus niveles
    DatossubType = Datos["subType"]
    LevelssubType=set(DatossubType)

    #Hacemos un bucle para cada nivel de categoria y se guardan las datos filtrados en un csv
    for province in LevelsRegion:
         for typebuild in LevelsType:
             for subTypebuild in LevelssubType:
                 DatosFiltrados= Datos[(Datos.type==typebuild)&(Datos.Region==province)&(Datos.subType==subTypebuild)]
                 #Un filtro para que no guarde csv vacios

                 if len(DatosFiltrados)!=0:
                    CsvFileName="DataOutput/"+str(province)+"_"+str(typebuild)+"_"+str(subTypebuild)+".csv"
                    print CsvFileName
                    DatosFiltrados.to_csv(CsvFileName,sep=";",encoding="UTF-8")
    
DatosRegion2CsvCategories("DataInput/DatosConComunidad.csv")