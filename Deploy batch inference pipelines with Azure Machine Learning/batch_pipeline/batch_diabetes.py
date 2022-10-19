
import os 
import numpy as np 
from azureml.core import Model 
import joblib 

def init():

    global model 

    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path) 

def run(mini_batch):

    resultList = []

    for f in mini_batch:
        data = np.genfromtxt(f, delimiter = ',')
        prediction = model.predict(data.reshape(1, -1))
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0])) 
    
    return resultList
