from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
import pickle
import numpy as np

app = Flask(__name__)
if CORS:
    CORS(app)
else:
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return response

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    features = [
        1 if data['gender'] == 'M' else 0,
        int(data['age']),
        int(data['smoking']),
        int(data['yellow_fingers']),
        int(data['anxiety']),
        int(data['peer_pressure']),
        int(data['chronic_disease']),
        int(data['fatigue']),
        int(data['allergy']),
        int(data['wheezing']),
        int(data['alcohol']),
        int(data['coughing']),
        int(data['shortness_of_breath']),
        int(data['swallowing_difficulty']),
        int(data['chest_pain'])
    ]
    
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    
    return jsonify({
        'prediction': 'YES' if prediction == 1 else 'NO',
        'probability': round(float(probability) * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)