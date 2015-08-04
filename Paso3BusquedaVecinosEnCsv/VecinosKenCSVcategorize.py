__author__ = 'david'
import pandas as pd

#Punto=pd.DataFrame([41.38465786,2.08005229,"Cataluna","house","Vacio"])
Punto=pd.DataFrame({'lat':[41.38465786],'lng':[2.08005229],'Region':["Cataluna"],'type':["house"],'subType':["Vacio"]})

print Punto
