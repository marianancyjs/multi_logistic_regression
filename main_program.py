import numpy as np
from flask import Flask, render_template, request
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model and encoder
with open('multi_logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the OneHotEncoder separately
with open('onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Home route to show the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    method = request.form['Activity']
    group = request.form['Group']
    
    # Encode the input using the saved encoder
    new_data = np.array([[method, group]])
    print(new_data)
    encoded_data = encoder.transform(new_data)
    print(encoded_data)

    # Make a prediction using the trained model
    probabilities = model.predict_proba(encoded_data)
    print(probabilities)

    # Prepare the result to display on the web page
    prob_class_0 = probabilities[0][0]
    prob_class_1 = probabilities[0][1]
    
    if prob_class_0>prob_class_1:
        v=0
    else:
        v=1

    return render_template('index.html', prob_class_0=prob_class_0,prob_class_1=prob_class_1)
                           

# Run the app
if __name__ == '__main__':
    app.run(debug=True)