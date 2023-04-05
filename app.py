from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Load the trained model from a pickle file
with open('model.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    problem_statement = "Problem Statement: Predicting Impressions from Digital Ad Campaigns"
    return render_template('test.html', problem_statement=problem_statement)


# Define the prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the HTML form
    campaign_type = int(request.form['campaign_type'])
    media_type = int(request.form['media_type'])
    traffic_source = int(request.form['traffic_source'])
    cost = float(request.form['cost'])

    # Make a prediction using the trained model
    X = np.array([[campaign_type, media_type, traffic_source, cost]])
    prediction = regressor.predict(X)

    # Return the prediction to the HTML template
    return render_template('results.html', prediction=prediction[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

