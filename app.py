from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    
    # Convert form data to the appropriate format for prediction
    data = [float(form_data[key]) for key in form_data]
    data = np.array(data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Determine risk
    risk = 'High Risk. Please consult a doctor.' if prediction[0] == 1 else 'Low Risk!'
    
    return render_template('result.html', risk=risk)

if __name__ == '__main__':
    app.run(debug=True)
