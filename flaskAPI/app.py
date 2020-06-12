# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:28:32 2020

@author: juan.alric
"""

import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle

app = Flask(__name__)


def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


@app.route('/predict', methods=['GET'])
def predict():
    # stub input features
    # parse input features from request
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1,-1)
    # load model
    model = load_models()
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)