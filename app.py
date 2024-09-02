from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form[feature]) for feature in ['CRIM', 'ZN', 'RM', 'AGE', 'DIS']]
        
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
