# # app.py
# from flask import Flask, request, jsonify
# from Predict.predcit.run import TextPredictionModel  # Adjust the import according to your project structure
#
# app = Flask(__name__)
#
# # Load model (adjust this according to your needs)
# model = TextPredictionModel.from_artefacts('../train/data/artefacts/2023-12-12-09-24-27/model.h5')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     text_list = data.get('texts', [])
#     top_k = data.get('top_k', 5)
#
#     predictions = model.predict(text_list, top_k=top_k)
#     return jsonify(predictions)
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
from run import TextPredictionModel

app = Flask(__name__)

model_dir = "../../train/data/artefacts/2024-01-09-15-54-23"

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def predict():
    model = TextPredictionModel.from_artefacts(model_dir)
    model_input = request.form['model_input']
    preds = model.predict([model_input], top_k=3)

    return preds

if __name__ == '__main__':
    app.run(debug=True)