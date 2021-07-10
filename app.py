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
    scaler.fit(inputs)
    inputs = scaler.transform(inputs)

    n_features = 1
    n_input = len(inputs)

    test_predictions = []

    first_eval_batch = inputs
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(0, 7):

        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]

        # append the prediction into the array
        test_predictions.append(current_pred)

        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [
                                  [current_pred]], axis=1)

    test_predictions = scaler.inverse_transform(test_predictions)
    test_predictions = [str(x[0]) for x in test_predictions]
    return Response(json.dumps(test_predictions))


if __name__ == '__main__':
    app.run(port=3004, debug=True)
