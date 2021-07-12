from flask import Flask, request, Response, json
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
app = Flask(__name__)


global model
model = keras.models.load_model("./modelRNN/rnn_enrgy_model.h5")


@app.route('/', methods=['GET'])
def hello_word():
    return "Hello world"


@app.route('/predictionsAPI', methods=['POST'])
def Predict():
    dataCSV = request.files['mydata']
    dataPath = "./data/" + dataCSV.filename
    dataCSV.save(dataPath)

    scaler = MinMaxScaler()

    inputs = pd.read_csv(dataPath, header=0,
                         infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    inputsConsumation = inputs[["Global_active_power", ]]
    inputsProduction = inputs[["Global_solar_production_power"]]
    print(inputs)
    print("-------------------------")
    scaler.fit(inputsConsumation)

    inputsConsumation = scaler.transform(inputsConsumation)
    inputsProduction = scaler.transform(inputsProduction)

    n_features = 1
    n_input = len(inputs)

    test_predictions_consummation = []
    test_predictions_production = []

    first_eval_batch_consummation = inputsConsumation
    current_batch_consummation = first_eval_batch_consummation.reshape(
        (1, n_input, n_features))
    # ---------------------------------
    first_eval_batch_production = inputsProduction
    current_batch_production = first_eval_batch_production.reshape(
        (1, n_input, n_features))

    for i in range(0, 7):

        # get the prediction value for the first batch
        current_pred_consummation = model.predict(
            current_batch_consummation)[0]
        current_pred_production = model.predict(current_batch_production)[0]

        # append the prediction into the array
        test_predictions_consummation.append(current_pred_consummation)
        test_predictions_production.append(current_pred_production)

        # use the prediction to update the batch and remove the first value
        current_batch_consummation = np.append(current_batch_consummation[:, 1:, :], [
            [current_pred_consummation]], axis=1)
        current_batch_production = np.append(current_batch_production[:, 1:, :], [
            [current_pred_production]], axis=1)

    test_predictions_consummation = scaler.inverse_transform(
        test_predictions_consummation)
    test_predictions_consummation = [
        str(x[0]) for x in test_predictions_consummation]

    test_predictions_production = scaler.inverse_transform(
        test_predictions_production)
    test_predictions_production = [
        str(x[0]) for x in test_predictions_production]

    jsonResponse = [test_predictions_consummation, test_predictions_production]

    return Response(json.dumps(jsonResponse))
