__author__ = 'David'
from geopy.geocoders import Nominatim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DatosTrain=pd.read_csv('data\DatosEntrenFiltrados.csv',index_col=0)
DatosTrain



geolocator = Nominatim()



location = geolocator.reverse("52.509669, 13.376294")
print(location.address)