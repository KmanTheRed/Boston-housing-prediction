from flask import Flask, jsonify, request
import pickle
import numpy as np
from flask_cors import CORS, cross_origin

# Initialize our Flask application
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the pre-trained model
model = pickle.load(open('housing.pickle', 'rb'))

@app.route("/")
@cross_origin()
def hello_world():
    return "Hello, cross-origin-world!"

@app.route("/predict", methods=["GET"])
def predict():
    if request.method == 'GET':
        # Create a dictionary to store the parameters
        params = {
            'CRIM': request.args.get('CRIM'),
            'ZN': request.args.get('ZN'),
            'INDUS': request.args.get('INDUS'),
            'CHAS': request.args.get('CHAS'),
            'NOX': request.args.get('NOX'),
            'RM': request.args.get('RM'),
            'AGE': request.args.get('AGE'),
            'DIS': request.args.get('DIS'),
            'RAD': request.args.get('RAD'),
            'TAX': request.args.get('TAX'),
            'PTRATIO': request.args.get('PTRATIO'),
            'LSTAT': request.args.get('LSTAT')
        }
        
        # Remove None values and convert remaining to float
        params = {k: float(v) for k, v in params.items() if v is not None}
        
        # Check if we have all required features
        required_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
        if all(feature in params for feature in required_features):
            final_features = [[params[feature] for feature in required_features]]
            prediction = model.predict(final_features)
            return jsonify({'prediction': prediction[0]})
        else:
            return jsonify({'error': 'Missing required features'}), 400
if __name__ == '__main__':
    app.run(debug=True)
