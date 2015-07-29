__author__ = 'David'

import numpy as np
import statsmodels.api as sm
import math
import pandas



data_df = pandas.read_csv('data\DatosEntrenamiento.csv')



ColPrecio = data_df['precio']

