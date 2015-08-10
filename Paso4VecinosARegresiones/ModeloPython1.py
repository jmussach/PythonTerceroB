# -*- coding: utf-8 -*-
"""
Machine learning functions for TerceroB

    - The functions similarity(), regression(), and train() do the real work
    - The functions similarity_main(), regression_main(), and train_main() just
    parse de arguments from the standard input, and then call the corresponding function. See
    comments in those functions for example calls

    - Additional assumptions: 
E        * About the MySQL schema: 

# Although the original file is 
#  [root@sandbox data2]# head -n 2 testigosALL.csv 
#  id,pid,lat,lng,price,"precio garaje","precio - gt","gross size","net size","land size",rooms,bedrooms,bathrooms,garaje,trastero,type,subType,planta,estado,terraceSize,community,energyPerformance,floorNumber,wardrobes,conditionedAir,heating,hotWater,caretaker,elevator,alarm,securityCams,reinforcedDoor,personalSecurity,tennisCourt,basketField,footballField,squash,pool,floorNumber
#  ilca5ae76f46,1,41.8613896,3.0713278,430000,0,426000,220,200,400,4,4,3,1,1,house,"terraced house",n.d.,"in good condition",,0,0.000,Si,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,n.d.,Si,n.d.
#  [root@sandbox data2]# 
    
        we use this schema where the first column is named pid of type varchar, and the second id of 
        type INT and primary key, because that second column is the one that is effectively a primary 
        key (there are repetitions for the value of the fist column), and id was the primary in the first
        data files we received. Anyway this is TBD
        
        * In the MySQL schema, in the column names we replace spaces by '_' because spaces are not
        allowed in Avro fields, and Avro is used as the serialization format for the import with Sqoop

        * train will be called with a data for the houses in a single polygon, type 
        and subtype, and will generate the parameters for the model, together with 
        the metrics for the model
        
        
        * to apply the model for an input house, the Lambdoop workflow will:
            1. Place the input house in a polygon according to its long and lat, and 
            in a type and subtype according to the corresponding fields
            2. Use this script to call similarity() for the input house and all the
            neighbours with the same polygon, type and subtype
            3. Take the m more similar neighbours among those evaluated in 2.
            4. Use this script to call regression() for the prices of the neighbours
            selected in 3. 
            
    - TODO: after this first version is developed, we should also consider an optimization
    algorithm that tries by joining together different polygons, in order to minimize the 
    errors of the produced models
"""

import json, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys, os, glob
import gc
import traceback
from functools import wraps
from multiprocessing import Process, Queue
import terceroB_ML_banks

_verbose_exceptions = True
# _verbose_exceptions = False

'''
Default value for  gc.get_threshold()
>>> gc.get_threshold()
(700, 10, 10)
'''
gc.set_threshold(70, 10, 10)

'''
If this works we should processify() is batches of n calls, to avoid overheads
'''
def processify(func):
    '''Decorator to run a function as a process.

    Be sure that every argument and the return value
    is *pickable*.

    The created process is joined, so the code does not
    run in parallel.

    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        p.join()
        ret, error = q.get()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper

lines_per_batch = 50

# _iterations_until_gc = 1000
_iterations_until_gc = 500
iterations_since_gc = 0
def update_garbage_collector():
    global iterations_since_gc
    if iterations_since_gc >= _iterations_until_gc:
        gc.collect()
        iterations_since_gc = 0
    iterations_since_gc += 1      
        
####################
# Similarity
####################
# @processify
def similarity(p1, p2):
    '''
    :param p1: dictionary with the data for a house
    :param p2: dictionary with the data for a house
        
    :returns: a float with the distance between the houses in the input. 
    NOTE: the bigger the number the less the similarity among p1 and p2
    
    Example: 
        In [8]: similarity({"lat" : 123123.2, "lng" : 12321.2, "gross_size" : 123312},
                   {"lat" : 234323.2, "lng" : 2321.2, "gross_size" : 23423})
        Out[8]: 149810.7216490195
    ''' 
    return terceroB_ML_banks.pos2dist(p1["lat"], p1["lng"], p2["lat"], p2["lng"])  

def similarity_main(): 
    '''
    Reads the arguments for similarity() from the standard input, and writes the
    result to the standard output.
    
    For each line in the input, we expect a string for a JSON document with two entries 
    p1, p2, where each pi is contains the data for a house. For each house we'll
    have a field per each column of the csv.

    For each input line we print not only the similarity but some data from the
    input that is needed in the next step of the Spark job, because the method 
    pipe() from RDDs doesn't allows us to preserve each command output paired
    with its corresponding input. This implies we interpret:
        - p1 is the house which price we are trying to predict
        - p2 is a neighbour
    
    Currently we ouput lines of the shape:
    
    <p1>|<similarity>|<p2["id"]>|<p2["price"]>|...
    
    NOTE: this corresponds to a database schema where id with type INT is the 
    primary key of the records
   
    Example call:
        
        { echo '{"p1" : {"id": "72006", "lat" : 123123.2, "lng" : 12321.2, "gross_size" : 123312, "price": "450000"}, "p2" : {"id": "72006", "lat" : 234323.2, "lng" : 2321.2, "gross_size" : 23423, "price": "550000"} }' ; 
          echo '{"p1" : {"id": "720021", "lat" : 123123.2, "lng" : 123212.2, "gross_size" : 13312, "price": "150000"}, "p2" : {"id": "32342", "lat" : 234323.2, "lng" : 2321.2, "gross_size" : 23423, "price": "150000"} }' ;  } | 
          python terceroB_ML.py similarity    
          
          {"lat": 123123.2, "lng": 12321.2, "price": "450000", "gross_size": 123312, "id": "72006"}|100039.120574|72006|550000
          {"lat": 123123.2, "lng": 123212.2, "price": "150000", "gross_size": 13312, "id": "720021"}|11403.4675333|32342|150000
    '''
    # lineno = 0
    for line in sys.stdin:
        # sys.stderr.write("Input line " + str(lineno) + " " + line + os.linesep)
        # lineno += 1
        args = json.loads(line) 
        # p1 is the house which price we are trying to predict; p2 is a neighbour
        p1, p2 = args["p1"], args["p2"]
        sim = similarity(p1, p2)
        sys.stdout.write('|'.join((
            json.dumps(p1), str(sim), 
            str(p2["id"]), str(p2["price"]), 
            str(p2["gross_size"]), str(p2["rooms"]), str(p2["planta"])
            )) + os.linesep)
        sys.stdout.flush()
        #sys.stderr.write("Done processing input line " + str(lineno) + os.linesep)
        #sys.stderr.flush()
        update_garbage_collector()
    
####################
# Regression
####################
# @processify
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
    rooms=p["rooms"]
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
    desvroomplus = 0
    desvroommin = 0
    desvplantaplus = 0
    desvplantamin = 0
    #genero las variables en desviaciones con sus medianas (si hace falta)
    if (pred_model == MODEL_2 or pred_model == MODEL_2P):
        
        desvm2=m2-np.median(nb_m2)
        if desvm2>0:
            desvm2plus=desvm2
        if desvm2<0:
            desvm2min=desvm2  

        desvroom=rooms-np.median(nb_rooms)
        if desvroom>0:
            desvroomplus=desvroom
        if desvroom<0:
            desvroommin=desvroom

        if (pred_model == MODEL_2P):
            desvplanta=planta-np.median(nb_plantas)
            if desvplanta>0:
                desvplantaplus=desvplanta
            if desvplanta<0:
                desvplantamin=desvplanta
        
    # Models with planta   
    if (pred_model == MODEL_1P or pred_model == MODEL_2P):

        pred_vars=np.column_stack((
            medianvec,m2,rooms,planta,
            medianvec*m2,
            medianvec*rooms,
            medianvec*planta,
            m2*rooms,
            m2*planta,
            rooms*planta))
        pred_final=np.dot(pred_vars,params["param_1p"])  
        
        if pred_model == MODEL_2P:
            
            pred_vars=np.column_stack((
                pred_final,
                desvm2plus,desvm2min,
                desvroomplus,desvroommin,
                desvplantaplus,desvplantamin))
            pred_final=np.dot(pred_vars,params["param_2p"]) 

    # Models without planta   
    else:
        pred_vars=np.column_stack((
            medianvec,m2,rooms,
            medianvec*m2,
            medianvec*rooms,
            m2*rooms))
        pred_final=np.dot(pred_vars,params["param_1"])  
    
        if pred_model == MODEL_2:
    
            pred_vars=np.column_stack((
                pred_final,
                desvm2plus,desvm2min,
                desvroomplus,desvroommin))
            pred_final=np.dot(pred_vars,params["param_2"]) 

    return pred_final
             
def regression_main(model=None):
    '''
    Reads the arguments for predict() from the standard input, and writes the
    result to the standard output.
    
    For each line in the input we expect 4 fields separated by "|" where:
     - field 0 is a string for a JSON document for the regression parameters as
     printed by train_main inside "predictions.model"
     - field 1 is a string for a JSON document with the data for an input house
     - field 2 is a string for a JSON list of neighbour ids
     - field 3 is a string for a JSON list of neighbour prices, in the same order as field 2
     
    For each input line we print not only the similarity but some data from the
    input that is needed in the next step of the Spark job, because the method 
    pipe() from RDDs doesn't allows us to preserve each command output paired
    with its corresponding input. Currently we ouput lines of the shape:
    
    <input_house["id"]>|<prediction>|<list of neighbour ids>
    
    :param model: use a model different to None to use a model from other module. Currently 
         only "banks" is supported to use terceroB_ML_banks.regression
    
    Example call:
    
        echo '{ "ratio2" : 0.5202176881785952 , "ratio1" : 66.17217407741025 , "ratio0" : 3.7664954121233016 , "param0" : [ -0.7804744393931945 , 3.353832287888097 , 1.8873176388954818 , 0.4148032632052172 , 7.388596835174136 , -11.121655010128805] , "param1" : [ 3.028115306513061 , -1.2547170218586232 , -1.6603740926074526E-5 , 1.1416762227369662E-5 , 0.0646383696136945 , -0.05570437256533732] , "param2" : [ 64.47336430982187 , -89.91549596740215 , -7.705408533464466E-5 , 1.3491690022418134E-4 , 1538732.5007242472 , 340023.60519339173 , -96527.3371590237 , -43902.41768630855 , -6664.60023244122 , -4420.799513257739] , "param3" : [ 64509.786425402446 , 58597.29920899326 , 48197.18879952711 , 53996.2809064412 , 5198.850206082088 , 280.46086681392853 , -336.5783340047509 , -17279.99192297754 , 346.48461406139535 , 222.9646962023994] , "ratio3" : 18.101643934241544}|{"basketField": "n.d.", "garaje": true, "bathrooms": 2, "conditionedAir": "n.d.", "caretaker": "n.d.", "wardrobes": "Si", "pid": "ilcc42db4f2e", "tennisCourt": "n.d.", "hotWater": "n.d.", "subType": "", "rooms": 4, "trastero": false, "lng": -3.687959, "land_size": 0, "id": 73091, "personalSecurity": "n.d.", "energyPerformance": 105.52, "footballField": "n.d.", "planta": 0, "elevator": "Si", "content": "Excelente piso esquinazo 4 dormitorios, ba\u00c3\u00b1o y aseo, amplio sal\u00c3\u00b3n exterior, calefacci\u00c3\u00b3n central, plaza de garaje. junto a plaza de castilla y estaci\u00c3\u00b3n de Chamart\u00c3\u00adn. rodeado de zonas verdes.", "type": "flat", "precio_garaje": 0, "reinforcedDoor": "n.d.", "price": 340000, "bedrooms": 4, "precio___gt": 340000, "squash": "n.d.", "gross_size": 118, "community": 0, "terraceSize": 0, "lat": 40.472095, "estado": "in good condition", "pool": "No", "alarm": "n.d.", "heating": "n.d.", "net_size": 110, "securityCams": "n.d.", "region": "almenara"}|[165755, 15012, 103429, 73091, 171252, 166422]|[410000, 280000, 200000, 340000, 480000, 325000]' |
            python terceroB_ML.py regression
    
        73091|30997618.862|[165755, 15012, 103429, 73091, 171252, 166422]
        
        
    echo '{ "ratio3" : 4.973448248811564 , "ratio2" : 5.507878210287414 , "param3" : [ 117700.88346965924 , -291775.151750442 , 427184.8484289724 , -48358.294432329596 , -9548.324433440226 , 2164.0659757362173 , -1658.039026445082 , -22195.02972506416 , 38024.6811449687 , 281.2508270886565] , "param2" : [ 1.3283081540773862 , -0.7271382723048896 , -5.469820808525801E-7 , 6.661060906355535E-7 , -395989.1595229766 , 577454.1865057369 , 99681.75278589809 , -32082.66810488065 , 2587.7514719718356 , -3277.9785497688445] , "param1" : [ 1.0371232500063394 , -0.12667415619959543 , -1.1533714498297775E-6 , 4.790315550399933E-7 , 0.005435589900298116 , -0.001969510263607164] , "param0" : [ 0.3063281755808191 , 0.31851857524053473 , 0.3164263229132284 , 0.33635555407627443 , -0.17615816397689793 , -0.05970964418684721] , "ratio1" : 6.940905667976217 , "ratio0" : 5.405950048984401}|{"basketField": "n.d.", "garaje": false, "bathrooms": 2, "conditionedAir": "n.d.", "caretaker": "n.d.", "wardrobes": "Si", "pid": "ilc1a620c51c", "tennisCourt": "n.d.", "community": 0, "subType": "", "rooms": 4, "trastero": false, "lng": -3.685329, "land_size": 0, "id": 153584, "personalSecurity": "n.d.", "energyPerformance": 0.0, "footballField": "n.d.", "elevator": "Si", "content": "VIVIENDA SITUADA FRENTE A LAS 4 TORRES, MUY LUMINOSO. TODOS LOS SERVICIOS. TIENE 100 M2., COCINA CON OFFICE, SAL\u00c3\u201cN, 4 DORMITORIOS, 2 BA\u00c3\u2018OS. CALIDADES: CARPINTERIA DE PVC, GOTELET, SAPELLI, GRES, GAS NATURAL. ORIENTACION SUR Y OESTE.FACIL APARCAMIENTO EN", "securityCams": "n.d.", "type": "flat", "reinforcedDoor": "n.d.", "price": 299000, "bedrooms": 4, "precio___gt": 299000, "squash": "n.d.", "gross_size": 100, "hotWater": "n.d.", "terraceSize": 0, "precio_garaje": 0, "estado": "in good condition", "pool": "No", "lat": 40.47242, "alarm": "n.d.", "heating": "n.d.", "net_size": 85, "planta": 0, "region": "castilla"}|[87281, 153584, 87279, 135343, 74848, 94075]|[300000, 299000, 370000, 159000, 490000, 175000]' | 
          python terceroB_ML.py regression
    ''' 
    reg_fun = terceroB_ML_banks.regression if (model == "banks") else regression
    # lineno = 0
    for line in sys.stdin:    
         #sys.stderr.write("Input line " + str(lineno) + " " + line + os.linesep)
         # lineno += 1
         category, regression_params, input_house, nb_ids, nb_prices, nb_m2, nb_rooms, nb_plantas = _regression_parse_input_line(line)
         prediction, status_code, maybe_except = _execute_regression(
             reg_fun, regression_params, input_house, nb_prices, nb_m2, nb_rooms, nb_plantas)      
         if _verbose_exceptions and status_code != _regression_ok_status_code:
             sys.stderr.write("regression exception " + str(maybe_except) + " for line " + line + os.linesep)                 
             sys.stderr.flush()             
         sys.stdout.write(_regression_build_output(input_house, prediction, status_code, nb_ids, category) + os.linesep)
         sys.stdout.flush()
         #sys.stderr.write("Done processing input line " + str(lineno) + os.linesep)
         #sys.stderr.flush()
         update_garbage_collector()
    
# value for the prediction in case of any error
_regression_error_prediction = -1 
# value of the status code for exceptions running a correct model
_regression_fail_run_status_code = -1
# value of the status code for regression abort due to faulty model
_regression_fail_faulty_model_status_code = -2
# value of the status code for successful execution of the regresion
_regression_ok_status_code = 0
def _execute_regression(regression, regression_params, input_house, nb_prices, nb_m2, nb_rooms, nb_plantas):
    '''
    Tries to execute the regression for the given params, input house and neighbours
    
    :returns: a triple (prediction, status_code, exception or None)
    '''
    if is_error_model(regression_params):
        return (_regression_error_prediction, _regression_fail_faulty_model_status_code, RuntimeError("faulty model"))
    try:
        prediction = regression(regression_params, input_house, nb_prices, nb_m2, nb_rooms, nb_plantas)
        return (prediction, _regression_ok_status_code, None)
    except Exception as e: 
        return (_regression_error_prediction, _regression_fail_run_status_code, e)
    
def _regression_parse_input_line(line):
    regression_params, input_house, nb_ids, nb_prices, nb_m2, nb_rooms, nb_plantas, category = line.split('|') 
    regression_params = json.loads(regression_params)
    input_house = json.loads(input_house)
    nb_prices = map(int, json.loads(nb_prices))
    nb_m2 = map(int, json.loads(nb_m2))
    nb_rooms = map(int, json.loads(nb_rooms))
    nb_plantas = map(int, json.loads(nb_plantas))
    return (
        category, regression_params, input_house, nb_ids, 
        nb_prices, nb_m2, nb_rooms, nb_plantas
    )
    
def _regression_build_output(input_house, prediction, status_code, nb_ids, category):
    return '|'.join([str(input_house["id"]), str(prediction), str(status_code), nb_ids, category])

####################
# Train
####################        

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
    price = np.array(input_data["price"], np.float)
    rooms = np.array(input_data["rooms"], np.float) 
    planta= np.array(input_data["planta"],np.float)
    
    dist=np.zeros((T,T))
    
    for i in range(T):
       dist[i][i]=float('Inf')
       for j in range(i+1, T):
           dist[i][j]=dist[j][i]=terceroB_ML_banks.pos2dist(lat[i],lgt[i],lat[j],lgt[j])

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
        medianvec,m2,rooms,
        medianvec*m2,
        medianvec*rooms,
        m2*rooms))

    # estimo los parámetros
    param_1=np.transpose(sm.OLS(priceTr,REG_1[:Tr,:]).fit().params)
    priceEstAll_1=np.dot(REG_1,param_1)
    
    # compruebo la bondad de los modelos
    errorpc_1=np.median(abs(priceTs-priceEstAll_1[Tr+1:T])/priceTs)
    
    # initialize with Inf
    errorpc_1p=float('Inf')
    
    if Splanta > 0:
    
        REG_1p=np.column_stack((
            medianvec,m2,rooms,planta,
            medianvec*m2,
            medianvec*rooms,
            medianvec*planta,
            m2*rooms,
            m2*planta,
            rooms*planta))

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
    desvroomplus=np.zeros((T))
    desvroommin=np.zeros((T))
    if Splanta > 0:
        desvplantaplus=np.zeros((T))
        desvplantamin=np.zeros((T))
    
    # genero las variables que miden asimetrias positivas frente a negativas
    for i in range(T):
        if desvm2[i]>0:
            desvm2plus[i]=desvm2[i]
        if desvm2[i]<0:
            desvm2min[i]=desvm2[i]  
        if desvroom[i]>0:
            desvroomplus[i]=desvroom[i]
        if desvroom[i]<0:
            desvroommin[i]=desvroom[i]
        if (Splanta > 0 and desvplanta[i]>0):
            desvplantaplus[i]=desvplanta[i]
        if (Splanta > 0 and desvplanta[i]<0):
            desvplantamin[i]=desvplanta[i]

    ####utilizo como regresores la previsión del modelo 1 y las desviaciones    
    REG_2=np.column_stack((
        priceEstAll_1,
        desvm2plus,desvm2min,
        desvroomplus,desvroommin))

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
            desvroomplus,desvroommin,
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
    
def groupkeys_to_fileName(groupkeys): 
    def clean(key):
        return re.sub('\s', '_', key).replace(',', "_or_")
    return "_".join(map(clean, groupkeys))

def train_main(from_csv=None, csv_sep=",", model=None):
    '''
    If from_csv is None, then reads the arguments for train() from the standard input, and writes the
    result to the standard output. Otherwise from_csv is a string path of an input csv with the test data.
    In both cases the model is trained and the formatted result is printed in the standard output

    We'll build a object pyInputLines : RDD[String] in Spark with for each category corresponding to a triple triple (region, type, subtype), and then perform a call to pyInputLines.pipe(...) per category. This achieves some distribution of the computation because different RDDs for different categories might be stored and processed in different slave nodes. This implies (to the best of my knowledge): 
        - For each partition in pyInputLines, Spark will call Python for the lines in that partition, which implies: 
                * the script might be called with no lines, for empty partitions in the RDD. In this case we just do nothing, printing nothing to stdout
                * the script might be called with more than one line. To respond to this we 
                treat each line separately, by training an independent model per each line. Hence each line must contain the whole training set for that model. This won't happen in practice because we have a different RDD per category, but this solution works anyway
        - We have to output a single line for each input line, otherwise different invocations of train() would be mixed in the output RDD returned by pipe(), which as any other RDD, it is not ordered
        - We get the values for (type, subtype) to call train from the first document in the line, as we'll call this script so all the documents in a single input line will have the same category. This means that any house with different values for (type, subtype) than those for the first record will be __filtered out of the training set__.  

    For these reasons we expect the following formats for the input and output: 
    - input: each line of the stdin will contain a JSON list of JSON objects, one object per house.
    - output: a call to train will be performed per each line in the input, and a single line will be printed to stdout for the result of the training. This output line will correspong to a JSON document, with a field "model" with a document for the model parameters, and a field "metrics" with a document for the metrics (see example below)

    Note that sys.argv[1] should be equal to "train" for the main of this file to route to this function1

    Note: use csv2jsons.py to convert test CSV data into the expected input 
    format. Note in production this records will be obtained from a query to MySQL
    
    :param model: use a model different to None to use a model from other module. Currently 
         only "banks" is supported to use terceroB_ML_banks.train
    
    Example: invocation (through this script's main)
    
    cat ../../../test/resources/python/ExportSomeWitnessesCategories_jsons/sabadell_flat_.json | python terceroB_ML.py train
{"house_type": "flat", "region": "sabadell", "predictions": {"metrics": {"error1pc": 0.21537656661982404, "error0pc": 0.2226621312163051, "error3pc": 0.2168584258813778, "error2pc": 0.17619141497698254}, "model": {"ratio3": 4.6113034157455477, "ratio2": 5.6756454344307237, "param3": [-11319.784938514966, 14098.379570575045, 8209.074124605333, 7101.478408260555, 188.38131002732462, 295.6474639312539, -4825.652647239414, 1683.2909464501008, 398.2969577287503, 71391.88809945964, 0.0, 0.0], "param2": [0.23839980731258884, 0.39874493483353035, -1.710405569573078e-07, -4.2398602668665786e-07, -4757.195310495405, -27091.7082394658, 10354.378917584963, 35015.84856306022, 20.940753616980025, 468.93535261299763, 40785.98913365419, 0.0, 0.0, 0.0], "param1": [0.3210320374158975, 0.569653130264353, -1.0491677510330873e-06, -9.727339832663755e-07, 0.0031448386805328302, 0.0013321156523063815, 10910.0317819449, 0.0, 0.0, 0.0], "param0": [0.18234724845027447, 0.18800382813123384, 0.15090413515162532, 0.13985600064293308, 0.1648573435464618, 0.12095887041873267, 11741.258910057339, 0.0, 0.0, 0.0], "ratio1": 4.6430306495003641, "ratio0": 4.4911094425326867}}, "house_subtype": ""}
    '''
    train_fun = terceroB_ML_banks.train if (model == "banks") else train
    
    if from_csv is not None:
        input_data_frame = pd.read_csv(from_csv, sep=csv_sep)
        #deprecated (format should be already in float witj "." as decimal separator)        
        #input_data_frame["lat"] = input_data_frame["lat"].apply(lambda s : float(str(s).replace(".","").replace(",","."))) / 1000000
        #input_data_frame["lng"] = input_data_frame["lng"].apply(lambda s : float(str(s).replace(".","").replace(",","."))) / 1000000       
        house_type, house_subtype, region = _extract_additional_train_input(input_data_frame)
        model_dict, metrics_dict = _execute_train(input_data_frame, house_type, house_subtype, train = train_fun) 
        print _train_build_output(model_dict, metrics_dict, region, house_type, house_subtype)
        return  
    
    for docs_list in sys.stdin:
        # NOTE this for implies we do nothing for calls with no lines
        # caused by empty partitions in Spark. 
        input_data_frame, house_type, house_subtype, region = _train_parse_input_doc_list(docs_list)
        model_dict, metrics_dict = _execute_train(input_data_frame, house_type, house_subtype, train = train_fun)      
        sys.stdout.write(_train_build_output(model_dict, metrics_dict, region, house_type, house_subtype) + os.linesep)
        sys.stdout.flush()
        update_garbage_collector()
       
def _train_parse_input_doc_list(docs_list):
    # Pandas expect a single JSON list with a JSON object per instance
    # NOTE: this loads the whole file in memory, but using the optimized
    # columnar format used by pandas 
    input_data_frame = pd.read_json(docs_list)
    # Get the type and subtype from the first element of the input data frame
    house_type, house_subtype, region = _extract_additional_train_input(input_data_frame)
    return (input_data_frame, house_type, house_subtype, region)
    
def _extract_additional_train_input(input_data_frame):
    # Get the type and subtype from the first element of the input data frame
    region = input_data_frame["region"][0]    
    house_type = input_data_frame["type"][0]
    house_subtype = input_data_frame["subType"][0]
    return (house_type, house_subtype, region)
    
def _train_build_output(model_dict, metrics_dict, region, house_type, house_subtype):
    '''
    NOTE: added field _model_status_field to indicate whether the model was faulty or not, with value:
        * _model_fail_status_code for faulty models, i.e. for which is_error_model() returns True
        * _model_ok_status_code for non faulty models
    '''
    model_status = _model_ok_status_code if not is_error_model(model_dict) else _model_fail_status_code
    model_category = groupkeys_to_fileName([region, house_type, house_subtype])
    return json.dumps({_model_status_field : model_status, _model_id_field : model_category,
                       "region" : region, "house_type" : house_type, "house_subtype" : house_subtype,
                       "predictions" : {"model" : model_dict, "metrics" : metrics_dict}})

_error_model = {} 
_error_model_metric = {} 
_model_status_field = "status"
_model_id_field = "category"
_model_fail_status_code = -2
_model_ok_status_code = 0
def is_error_model(model):
    return model == _error_model
    
def _execute_train(input_data_frame, house_type, house_subtype, train=train):
    try: 
        (model_dict, metrics_dict) = train(input_data_frame, house_type=house_type, house_subtype=house_subtype)
    except Exception as e:
        if _verbose_exceptions:
            sys.stderr.write("train exception: " + str(e) + os.linesep)
            sys.stderr.flush()             
        model_dict = _error_model
        metrics_dict = _error_model_metric
    return (model_dict, metrics_dict)
    
def json2csv(input_path, fields, quote='', sep=','):
    '''
    Transform a json or jsons file into csv a file. Input formats expeced: 
        - json: a single JSON list of JSON documents
        - jsons: one JSON document per line 
        
    As a result a file at input_path but replacing the extension with .csv is created
    such as:
        - the first line is the header, corresponding to joining fields with quote and sep
        - a line is generated per document in the input, by getting the fields specified 
        in fields from the document. This only workds for first level fields. If some
        document is missing any field then an exception is raised
        
    :param fields: string with comma separated names of the fields to use in the output. Expected only
    first level fields. Also fields will be added as a header as the first line    
    
    Example calls (from main):
    
    python terceroB_ML.py json2csv /mnt/data_home/git/lambdoop-pilots/tercerob/tercerob-pricing/src/test/resources/python/ExportSomeWitnessesCategories_jsons/sabadell_flat_.json id,pid,lat,lng,price,"precio garaje","precio - gt","gross size","net size","land size",rooms,bedrooms,bathrooms,garaje,trastero,type,subType,planta,estado,terraceSize,community,energyPerformance,wardrobes,conditionedAir,heating,hotWater,caretaker,elevator,alarm,securityCams,reinforcedDoor,personalSecurity,tennisCourt,basketField,footballField,squash,pool,region
    python terceroB_ML.py json2csv /mnt/data_home/git/lambdoop-pilots/tercerob/tercerob-pricing/src/test/resources/testigoscastilla.bak.jsons trastero,bathrooms
    '''
    fields = [re.sub("\s|-", "_", field) for field in fields.split(",")]
    # load the data
    try: 
        # try interpreting the input as a single JSON list of documents
        with open(input_path, 'r') as in_f:
            data = json.load(in_f)
    except:
        # now try interpreting the input as one JSON object per line
        data = []
        with open(input_path, 'r') as in_f:
            for line in in_f:
                data.append(json.loads(line))
        
    # now data is an iterable of documents
    output_path = os.path.splitext(input_path)[0] + ".csv"
    with open(output_path, 'w') as out_f:
        out_f.write(sep.join((quote + field + quote for field in fields)))
        out_f.write(os.linesep)
        for doc in data:             
            out_f.write(sep.join((quote + str(doc[field]) + quote for field in fields)) + os.linesep)
    print "Result file can be found at", output_path
    
def model_from_hdfs_to_csv_report(input_path, output_path):
    '''
    :param input_path: path to a director containing the part-* files corresponding to 
    a get of the models generated in HDFS by the Lambdoop workflow
    '''
    error_metrics = [
        "errorpc_1", "errorpc_1p", "errorpc_2", "errorpc_2p", 
        "size", "type", "subt", "train", "test"]
    parameters = ["param_1", "param_1p", "param_2", "param_2p", "index_err"]
    header = ["category", "status"] + error_metrics + parameters
    with open(output_path, 'w') as out_f:
        out_f.write(','.join(header) + os.linesep)
        for path in glob.iglob(os.path.join(input_path, "part-*")): 
            with open(path, 'r') as in_f:
                for line in in_f:
                    model_doc = json.loads(line)
                    '''
                    See _train_build_output(model_dict, metrics_dict, region, house_type, house_subtype):
                    
                    return json.dumps({"predictions" : {"model" : model_dict, "metrics" : metrics_dict}, 
                                       "region" : region, "house_type" : house_type, "house_subtype" : house_subtype,
                                       _model_status_field : _model_status,  _model_id_field : model_category
                                       })
                                       
                    Error models are like: 
                    
                    { "status" : -2 , "category" : "sant_juliá_del_llor_i_bonmati_flat_" , 
                    "region" : "sant juliá del llor i bonmati" ,
                      "predictions" : { "metrics" : { } , "model" : { }} , "house_type" : "flat" , "house_subtype" : ""}

                    '''                                    
                    row = ','.join([ model_doc[_model_id_field], str(model_doc[_model_status_field]) ] + 
                                   [ str(model_doc["predictions"]["metrics"].get(metric)) for metric in error_metrics ] + 
                                   [ str(model_doc["predictions"]["model"].get(param)).replace(",","") for param in parameters ] )
                    out_f.write(row.encode('utf-8') + os.linesep)

if __name__ == '__main__':        
    if len(sys.argv) < 2: 
        print 'Usage:', sys.argv[0], '[similarity | regression | train | train_csv | json2csv]'
        print 'the script will read the rest of the input data from stdin'
        sys.exit(1)
    action = sys.argv[1]
    if action == 'similarity':
        similarity_main()
    elif action == 'regression':
        model = sys.argv[2] if len(sys.argv) >= 3 else None
        regression_main(model=model)
    elif action == 'train':
        model = sys.argv[2] if len(sys.argv) >= 3 else None
        train_main(model=model)
    elif action == 'train_csv':
        '''
        Example: 
            python terceroB_ML.py train_csv /mnt/data_home/git/lambdoop-pilots/tercerob/tercerob-pricing/src/test/resources/python/ExportSomeWitnessesCategories_jsons/sabadell_flat_.bak.csv
        '''
        try: 
            path = sys.argv[2]
            model = sys.argv[3] if len(sys.argv) >= 4 else None
            train_main(model=model, from_csv=path)
        except Exception as e: 
            print "Exception training model:", e
            print 'Usage', sys.argv[0], "train_csv", "<input path>"
    elif action == "json2csv":
        '''
        Example: 
            python terceroB_ML.py json2csv /mnt/data_home/git/lambdoop-pilots/tercerob/tercerob-pricing/src/test/resources/python/ExportSomeWitnessesCategories_jsons/sabadell_flat_.bak.json id,pid,lat,lng,price,"precio garaje","precio - gt","gross size","net size","land size",rooms,bedrooms,bathrooms,garaje,trastero,type,subType,planta,estado,terraceSize,community,energyPerformance,floorNumber,wardrobes,conditionedAir,heating,hotWater,caretaker,elevator,alarm,securityCams,reinforcedDoor,personalSecurity,tennisCourt,basketField,footballField,squash,pool,floorNumber,region
        '''
        try: 
            json2csv(sys.argv[2], sys.argv[3])
        except Exception as e: 
            print "Exception converting from JSON to CSV:", e
            print 'Usage', sys.argv[0], "json2csv", "<input path>", "<fields>"
    elif action == "generate_report":
        '''
        Example: 
            python terceroB_ML.py generate_report /mnt/data_home/TerceroB/Febrero2015/models/latest report.csv
        '''
        try:
            model_from_hdfs_to_csv_report(sys.argv[2], sys.argv[3])
        except Exception as e: 
            print "Exception generating report:", e
            print 'Usage', sys.argv[0], "generate_report", "<input path>", "<output path>"
    else:
        sys.stderr.write('wrong action' + os.linesep)
        sys.exit(2)      
