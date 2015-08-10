from shapely.geometry import shape,Point
import pygeoj
import pandas as pd
import numpy as np

DatosAValorar=pd.read_csv(DatosSinFiltrar,sep=";",index_col=0,error_bad_lines=False)