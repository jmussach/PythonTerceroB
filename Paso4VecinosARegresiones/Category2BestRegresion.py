__author__ = 'david'
# coding=utf-8
from VecinosKenCSVcategorize import punto2vecinosk
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn.neighbors as sk

DatosAValorar=pd.read_csv("DataInput/DatosAValorarConRegion.csv",sep=";",index_col=0,encoding="UTF-8")

DatosAValorar.iloc[0]
#Vecinos=punto2vecinosk(InmuebleAValorar,6)

PuntoDic=DatosAValorar.iloc[0]
CSVdireccion="DataCategorize/"+PuntoDic['Region']+"_"+PuntoDic['type']+"_"+PuntoDic['subType']+".csv"
Datos=pd.read_csv(CSVdireccion,sep=";",index_col=0)

LatLong=Datos.loc[:,['lat','lng']].values
#Calculamos la distancia de todos con todos para cada categoria BallTree con Haversine
tree =sk.BallTree(LatLong,metric="haversine")#Calcula la distancia Habersine de todos los puntos entre si
#y los ordena en estructura de arbol
MedianVec=[]
for i in range(len(LatLong)):
    dist, ind = tree.query(LatLong[i], k=6)

    MedianVec.append(np.median(Datos.iloc[ind[0]]['price']))


T=len(Datos)
Tr=round(0.75*T) #Muestra de entrenamiento

#Datos con los Precios de entrenamiento y de validacion
priceTr=Datos.iloc[0:Tr]['price']
priceTs=Datos.iloc[Tr+1:T]['price']

#Suma de los datos de planta de los datos de entrenamiento
Splanta=np.sum(Datos.iloc[0:Tr]['planta'])


#genero las variables en desviaciones con sus medianas:
#Desv m2:

desvm2=Datos.loc[:,'gross.size'].values-np.median(Datos.iloc[ind[0]]['gross.size'])
#desvroom=rooms-np.median(rooms[index_vecinos], axis=1)
desvroom=Datos.loc[:,'rooms'].values-np.median(Datos.iloc[ind[0]]['rooms'])

if Splanta > 0:
    desvplanta=Datos.loc[:,'planta'].values-np.median(Datos.iloc[ind[0]]['planta'])
else:
    desvplanta=0

#genero la primera matriz de regresores donde utilizamos la mediana de los
#seis vecinos las variables hedonicas y sus interacciones


REG_1=np.column_stack((MedianVec,
                       Datos.loc[:,'gross.size'].values,
                       Datos.loc[:,'rooms'].values,
                       MedianVec*Datos.loc[:,'gross.size'].values,
                       MedianVec*Datos.loc[:,'rooms'].values,
                       Datos.loc[:,'gross.size'].values*MedianVec*Datos.loc[:,'rooms'].values))



# # estimo los parametros
param_1=sm.OLS(priceTr,REG_1[:Tr,:]).fit().params
priceEstAll_1=np.dot(REG_1,param_1)
#
# # compruebo la bondad de los modelos
errorpc_1=np.median(abs(priceTs-priceEstAll_1[Tr+1:T])/priceTs)

# initialize with Inf
errorpc_1p=float('Inf')

if Splanta > 0:

    REG_1p=np.column_stack((MedianVec,
                       Datos.loc[:,'gross.size'].values,
                       Datos.loc[:,'rooms'].values,
                       Datos.loc[:,'planta'].values,
                       MedianVec*Datos.loc[:,'gross.size'].values,
                       MedianVec*Datos.loc[:,'rooms'].values,
                       MedianVec*Datos.loc[:,'planta'].values,
                       Datos.loc[:,'gross.size'].values*MedianVec*Datos.loc[:,'rooms'].values,
                       Datos.loc[:,'gross.size'].values*MedianVec*Datos.loc[:,'planta'].values,
                       Datos.loc[:,'planta'].values*MedianVec*Datos.loc[:,'rooms'].values))


    # estimo los parametros
    param_1p=sm.OLS(priceTr,REG_1p[:Tr,:]).fit().params
    priceEstAll_1p=np.dot(REG_1p,param_1p)

    # compruebo la bondad de los modelos
    errorpc_1p=np.median(abs(priceTs-priceEstAll_1p[Tr+1:T])/priceTs)

###########################################################################
# Adjusted models
###########################################################################

# para los siguientes modelos, tengo que trabajar con los datos
desvm2plus = np.zeros((T))
desvm2min = np.zeros((T))
desvroomplus = np.zeros((T))
desvroommin = np.zeros((T))
if Splanta > 0:
    desvplantaplus = np.zeros((T))
    desvplantamin = np.zeros((T))

# genero las variables que miden asimetrias positivas frente a negativas
for i in range(T):
    if desvm2[i] > 0:
        desvm2plus[i] = desvm2[i]
    if desvm2[i] < 0:
        desvm2min[i] = desvm2[i]
    if desvroom[i] > 0:
        desvroomplus[i] = desvroom[i]
    if desvroom[i] < 0:
        desvroommin[i] = desvroom[i]
    if (Splanta > 0 and desvplanta[i] > 0):
        desvplantaplus[i] = desvplanta[i]
    if (Splanta > 0 and desvplanta[i] < 0):
        desvplantamin[i] = desvplanta[i]

    ####utilizo como regresores la prevision del modelo 1 y las desviaciones
REG_2=np.column_stack((priceEstAll_1,
    desvm2plus,desvm2min,
    desvroomplus,desvroommin))

param_2=sm.OLS(priceTr,REG_2[:Tr,:]).fit().params
priceEstTs_2=np.dot(REG_2[Tr+1:T,:],param_2)


# compruebo la bondad de los modelos
errorpc_2=np.median(abs(priceTs-priceEstTs_2)/priceTs)
print errorpc_2
# initialize with Inf
#errorpc_2p=float('Inf')

# initialize with Inf
errorpc_2p=float('Inf')

if Splanta > 0:

    REG_2p=np.column_stack((
        priceEstAll_1p,
        desvm2plus,desvm2min,
        desvroomplus,desvroommin,
        desvplantaplus,desvplantamin))

    param_2p=sm.OLS(priceTr,REG_2p[:Tr,:]).fit().params
    priceTsEstTs_2p=np.dot(REG_2p[Tr+1:T,:],param_2p)

    # compruebo la bondad de los modelos
    errorpc_2p=np.median(abs(priceTs-priceTsEstTs_2p)/priceTs)

###########################################################################
# Build the return value, which is a pair of dictionaries:
###########################################################################

#ordeno los errores de menor a mayor
index_err=np.array([errorpc_1,errorpc_1p,errorpc_2,errorpc_2p]).argsort()
param_models=np.array([param_1.tolist(),param_1p.tolist(),param_2.tolist(), param_2p.tolist()])
param_models[index_err[0]]
#  the first one is a dictionary for the output model
if Splanta > 0:
    model_dict = {"param_1" : param_1.tolist(), "param_1p": param_1p.tolist(),
                  "param_2" : param_2.tolist(), "param_2p": param_2p.tolist(),
                  "index_err": index_err.tolist()}
else:
    model_dict = {"param_1" : param_1.tolist(), "param_1p": [],
                  "param_2" : param_2.tolist(), "param_2p": [],
                  "index_err": index_err.tolist()}

#  the second one is a dictionary for the error metrics
metrics_dict = {"errorpc_1" : errorpc_1, "errorpc_1p" : errorpc_1p,
                "errorpc_2" : errorpc_2, "errorpc_2p" : errorpc_2p,
                "size" : len(Datos),
                "train" : Tr, "test" : (T-Tr),
                "Region" :PuntoDic['Region'],
                "type" : PuntoDic['type'],
                "subt" : PuntoDic['subType']}


print param_models[index_err[0]]
#Los parametros del error mas pequeno
# print model_dict.get("param_1")
# print errorpc_1,errorpc_1p,errorpc_2,errorpc_2p
print "###"
print param_2.tolist()
# print "###"
# print model_dict

# print metrics_dict
