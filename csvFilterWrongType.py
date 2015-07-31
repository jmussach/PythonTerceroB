__author__ = 'David'
import csv
with open('data/DatosEntrenamiento.csv', 'rb') as csvfile:
    Archivo= csv.reader(csvfile, delimiter=',', quotechar='|')
    ArchivoList=list(Archivo)
    print ArchivoList[0:10][4:5]