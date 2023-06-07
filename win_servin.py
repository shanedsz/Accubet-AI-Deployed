from flask import Flask, request, jsonify
import pickle
import numpy as np


def predict_single(test, dv, model):
    X = dv.transform([test])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('win-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('win')

@app.route('/predict', methods=['POST'])
def predict():
    test = request.get_json()

    prediction = predict_single(test, dv, model)
    win = prediction >= 0.6

    result = {
        'win_probability' : float(prediction),
        'win' : bool(win) 
    }


if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=9696)