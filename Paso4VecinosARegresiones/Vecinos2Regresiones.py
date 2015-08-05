__author__ = 'david'
# -*- coding: utf-8 -*-
from VecinosKenCSVcategorize import punto2vecinosk
import pandas as pd

InmuebleAValorar={'lat':[41.38465786],'lng':[2.08005229],'Region':["Cataluña"],'type':["house"],'subType':["Vacio"]}
Vecinos=punto2vecinosk(InmuebleAValorar,15)

PuntoDic=pd.DataFrame(InmuebleAValorar)
CSVdireccion="DataCategorize/"+PuntoDic.iloc[0]['Region']+"_"+PuntoDic.iloc[0]['type']+"_"+PuntoDic.iloc[0]['subType']+".csv"
Datos=pd.read_csv(CSVdireccion,sep=";",index_col=0)

T=len(Datos)
Tr=round(0.75*T) #Muestra de entrenamiento

priceTr=Datos.iloc[0:Tr]['price']
priceTs=Datos.iloc[Tr+1:T]['price']

# [JB] Only take into account the training samples for "planta"
#Splanta=np.sum(planta[0:Tr])
print priceTs