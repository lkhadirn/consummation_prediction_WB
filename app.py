from flask import Flask, request, Response, json
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_word():

    return "hello world"


@app.route('/predictionsAPI', methods=['POST'])
def Predict():
    dataCSV = request.files['mydata']
    dataPath = "./data/" + dataCSV.filename
    dataCSV.save(dataPath)

    inputs = pd.read_csv("data/testData.csv")
    inputs = np.array(inputs["0"].values)
    global model
    model = keras.models.load_model('C:/Users/dell/Desktop/output')
    # scallingFit = pd.read_csv('C:/Users/dell/Desktop/scallingFit.csv')
    # scallingFit = np.array(inputs["0"].values)
    # sc.fit_transform(scallingFit)
    # We need to Reshape
    inputs = inputs.reshape(-1, 1)
    # Normalize the Dataset
    #inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 160):
        X_test.append(inputs[i-60:i])

    # Convert into Numpy Array
    X_test = np.array(X_test)
    # Reshape before Passing to Network
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted = model.predict(X_test)
    # Do inverse Transformation to get Values
    #predicted_MG = sc.inverse_transform(predicted)
    predicted = [str(x[0]) for x in predicted]

    return Response(json.dumps(predicted[:7]))


if __name__ == '__main__':
    app.run(port=3004, debug=True)
