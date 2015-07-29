# -*- coding: utf-8 -*-
"""
Models for banks
"""
import numpy as np
import statsmodels.api as sm
import math

# from terceroB_ML import pos2dist
def pos2dist(lat1,long1,lat2,long2):  
    """ 
    Calcula distancia entre dos puntos conociendo latitud y longitud 
    sacado de la fórmula de Haversine (http://en.wikipedia.org/wiki/Haversine_formula). 
    radius=6371;
    lat1=latlon1(1)*pi/180;
    lat2=latlon2(1)*pi/180;
    lon1=latlon1(2)*pi/180;
    lon2=latlon2(2)*pi/180;
    deltaLat=lat2-lat1;
    deltaLon=lon2-lon1;
    a=sin((deltaLat)/2)^2 + cos(lat1)*cos(lat2) * sin(deltaLon/2)^2;
    c=2*atan2(sqrt(a),sqrt(1-a));
    %d1km=radius*c;    %Haversine distance
    
    x=deltaLon*cos((lat1+lat2)/2);
    y=deltaLat;
    d=radius*sqrt(x*x + y*y); %Pythagoran distance
    """  
    lat1 = math.pi*lat1/180
    lat2 = math.pi*lat2/180
    long1 = math.pi*long1/180
    long2 = math.pi*long2/180
    d_latt = lat2 - lat1  
    d_long = long2 - long1  
    #a = math.sin(d_latt/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long/2)**2  
    #c = 2 * math.asin(math.sqrt(a))  
    x=d_long*math.cos((lat1+lat2)/2)
    y=d_latt
    c=math.sqrt(x**2+y**2)
    return 6371 * c  

def regression(params, p, nb_prices, nb_m2, nb_rooms, nb_plantas):
    '''
    :param params: dictionary with the parameters of the regression. 
    
    :param p: dictionary with the data for a house. Example:
    
    {"lat" : 123123.2, "lng" : 12321.2, "gross_size" : 123312, "bathrooms" : 12, "rooms" : 4, "trastero" : 1, "garaje" : 0, "type" : "flat", "price" : 1231}    
    
    :param neighbour_prices: list of floats with the prices for the 
    neighbours of p. Examples: 
    
    [31312, 123123, 654654, 324324, 23434, 423423]
        
    :returns: a float with the predicted price
    
    '''
    m2=p["gross_size"]   
    planta=p["planta"]

    #caLculo precio mediano de los vecinos del piso a valorar
    medianvec=np.median(nb_prices)  

    # Check which model to use
    MODEL_1, MODEL_1P, MODEL_2, MODEL_2P = 0, 1, 2, 3
    index_err = params["index_err"]

    # By deafult first ranked model is chosen
    pred_model = index_err[0]
   
    # If planta is missing use a model that is not based on it
    if (planta == 0 and pred_model != MODEL_1 and pred_model != MODEL_2):
        # discard planta-based model and use 2nd ranked model
        pred_model = index_err[1]
        if (pred_model != MODEL_1 and pred_model != MODEL_2):
            # discard planta-based model and use 3rd ranked model
            pred_model = index_err[2]

    desvm2plus = 0
    desvm2min = 0
    desvplantaplus = 0
    desvplantamin = 0
    #genero las variables en desviaciones con sus medianas (si hace falta)
    if (pred_model == MODEL_2 or pred_model == MODEL_2P):
        
        desvm2=m2-np.median(nb_m2)
        if desvm2>0:
            desvm2plus=desvm2
        if desvm2<0:
            desvm2min=desvm2

        if (pred_model == MODEL_2P):
            desvplanta=planta-np.median(nb_plantas)
            if desvplanta>0:
                desvplantaplus=desvplanta
            if desvplanta<0:
                desvplantamin=desvplanta
        
    # Models with planta   
    if (pred_model == MODEL_1P or pred_model == MODEL_2P):
    
        pred_vars=np.column_stack((
            medianvec,m2,planta,
            medianvec*m2,
            medianvec*planta,
            m2*planta))
        pred_final=np.dot(pred_vars,params["param_1p"])  
        
        if pred_model == MODEL_2P:
            
            pred_vars=np.column_stack((
                pred_final,
                desvm2plus,desvm2min,
                desvplantaplus,desvplantamin))
            pred_final=np.dot(pred_vars,params["param_2p"]) 

    # Models without planta   
    else:
        pred_vars=np.column_stack((
            medianvec,m2,
            medianvec*m2))
        pred_final=np.dot(pred_vars,params["param_1"])  
    
        if pred_model == MODEL_2:
    
            pred_vars=np.column_stack((
                pred_final,
                desvm2plus,desvm2min))
            pred_final=np.dot(pred_vars,params["param_2"]) 

    return pred_final


def train(input_data, house_type="flat", house_subtype=""):
    '''
    :param input_data: Pandas data frame with the input data
    :param house_type: string with the type of the houses corresponding to 
    this model, houses in input_data with other type will be discarded
    :param house_subtype: currently not used, in the future will be used
    like house_type
    
    :returns a pair of dictionaries:
        * the first one is a dictionary for the output model
        * the second one is a dictionary for the error metrics
    '''    

    original_size = len(input_data)
    
    # Data load    
    #   Drop houses which are not flats
    #   NOTE: originally the encoding flat = 1, house = 2, office = 3 was used
    #   that encoding was probably originated by an enumeration performed by numpy
    input_data = input_data[input_data["type"] == house_type]
    type_size = len(input_data)
    if (len(house_subtype) > 0):
        input_data = input_data[input_data["subType"] == house_subtype]
    subtype_size = len(input_data)

    T = len(input_data)
    
    #   latitud y longitud tienen que ser números decimales (ojo con el formato idealista)
    lat   = np.array(input_data["lat"], np.float) 
    lgt   = np.array(input_data["lng"], np.float)  
    m2    = np.array(input_data["gross_size"], np.float)
    rooms = np.array(input_data["rooms"], np.float) 
    price = np.array(input_data["price"], np.float)
    planta= np.array(input_data["planta"],np.float)
    
    dist=np.zeros((T,T))
    
    for i in range(T):
       dist[i][i]=float('Inf')
       for j in range(i+1, T):
           dist[i][j]=dist[j][i]=pos2dist(lat[i],lgt[i],lat[j],lgt[j])

    ##ahora ordenamos la matriz por vecindades
    n_vecinos=min(T-1, 6)
    index_vecinos=dist.argsort(axis=1)[:,0:n_vecinos]
    
    ############################# SAMPLE ###########################

    Tr=round(0.75*T) #Muestra de entrenamiento
    
    priceTr=price[0:Tr]
    priceTs=price[Tr+1:T]
    
    # [JB] Only take into account the training samples for "planta"
    Splanta=np.sum(planta[0:Tr])
    
    ############################ MEDIANAS ###########################
    
    #obtengo la mediana de los seis vecinos
    medianvec=np.median(price[index_vecinos], axis=1)

    #genero las variables en desviaciones con sus medianas
    desvm2=m2-np.median(m2[index_vecinos], axis=1)
    desvroom=rooms-np.median(rooms[index_vecinos], axis=1)
    if Splanta > 0:
        desvplanta=planta-np.median(planta[index_vecinos], axis=1)
    else:
        desvplanta=0
        
    ###########################################################################
    # Base models
    ###########################################################################
    
    #genero la primera matriz de regresores donde utilizamos la mediana de los 
    #seis vecinos las variables hedonicas y sus interacciones
    
    REG_1=np.column_stack((
        medianvec,m2,
        medianvec*m2))

    # estimo los parámetros
    param_1=np.transpose(sm.OLS(priceTr,REG_1[:Tr,:]).fit().params)
    priceEstAll_1=np.dot(REG_1,param_1)
    
    # compruebo la bondad de los modelos
    errorpc_1=np.median(abs(priceTs-priceEstAll_1[Tr+1:T])/priceTs)
    
    # initialize with Inf
    errorpc_1p=float('Inf')
    
    if Splanta > 0:
    
        REG_1p=np.column_stack((
            medianvec,m2,planta,
            medianvec*m2,
            medianvec*planta,
            m2*planta))

        # estimo los parámetros
        param_1p=np.transpose(sm.OLS(priceTr,REG_1p[:Tr,:]).fit().params)
        priceEstAll_1p=np.dot(REG_1p,param_1p)
        
        # compruebo la bondad de los modelos
        errorpc_1p=np.median(abs(priceTs-priceEstAll_1p[Tr+1:T])/priceTs)
    
    ###########################################################################
    # Adjusted models
    ###########################################################################
    
    # para los siguientes modelos, tengo que trabajar con los datos
    desvm2plus=np.zeros((T))
    desvm2min=np.zeros((T))
    if Splanta > 0:
        desvplantaplus=np.zeros((T))
        desvplantamin=np.zeros((T))
    
    # genero las variables que miden asimetrias positivas frente a negativas
    for i in range(T):
        if desvm2[i]>0:
            desvm2plus[i]=desvm2[i]
        if desvm2[i]<0:
            desvm2min[i]=desvm2[i]  
        if (Splanta > 0 and desvplanta[i]>0):
            desvplantaplus[i]=desvplanta[i]
        if (Splanta > 0 and desvplanta[i]<0):
            desvplantamin[i]=desvplanta[i]

    ####utilizo como regresores la previsión del modelo 1 y las desviaciones    
    REG_2=np.column_stack((
        priceEstAll_1,
        desvm2plus,desvm2min))

    param_2=np.transpose(sm.OLS(priceTr,REG_2[:Tr,:]).fit().params)
    priceEstTs_2=np.dot(REG_2[Tr+1:T,:],param_2)
    
    # compruebo la bondad de los modelos
    errorpc_2=np.median(abs(priceTs-priceEstTs_2)/priceTs)    
    
    # initialize with Inf
    errorpc_2p=float('Inf')
    
    if Splanta > 0:
    
        REG_2p=np.column_stack((
            priceEstAll_1p,
            desvm2plus,desvm2min,
            desvplantaplus,desvplantamin))

        param_2p=np.transpose(sm.OLS(priceTr,REG_2p[:Tr,:]).fit().params)
        priceTsEstTs_2p=np.dot(REG_2p[Tr+1:T,:],param_2p)
        
        # compruebo la bondad de los modelos
        errorpc_2p=np.median(abs(priceTs-priceTsEstTs_2p)/priceTs)        
    
    ###########################################################################
    # Build the return value, which is a pair of dictionaries:
    ###########################################################################

    #ordeno los errores de menor a mayor
    index_err=np.array([errorpc_1,errorpc_1p,errorpc_2,errorpc_2p]).argsort()
 
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
                    "size" : original_size, 
                    "type" : type_size, "subt" : subtype_size,
                    "train" : Tr, "test" : (T-Tr)}
                    
    return (model_dict, metrics_dict) 
