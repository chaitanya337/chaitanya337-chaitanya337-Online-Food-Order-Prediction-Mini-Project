
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: 'model.pkl' file not found. Ensure the model file is in the same directory as app.py.")

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        marital_status = int(request.form['marital_status'])
        occupation = int(request.form['occupation'])
        monthly_income = int(request.form['monthly_income'])
        educational_qualifications = int(request.form['educational_qualifications'])
        family_size = float(request.form['family_size'])
        pin_code = float(request.form['pin_code'])
        feedback = int(request.form['feedback'])

        # Create a feature array
        features = np.array([[age, gender, marital_status, occupation, 
                              monthly_income, educational_qualifications, 
                              family_size, pin_code, feedback]])

        # Make a prediction
        prediction = model.predict(features)

        # Map the prediction to output
        output = "Yes" if prediction[0] == 1 else "No"

    except KeyError as e:
        # Handle missing form fields
        return f"Form field {e.args[0]} is missing. Please fill out all fields.", 400
    except Exception as e:
        # Handle other exceptions, like model errors
        return f"An error occurred: {str(e)}", 500

    return render_template('index.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
