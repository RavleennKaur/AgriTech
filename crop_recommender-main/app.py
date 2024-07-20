from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model

# Create the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('my_model.h5')
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Fit the scaler and label encoder
data = pd.read_csv('./r.csv')
feature_names = ['temperature', 'humidity', 'ph', 'rainfall', 'N', 'P', 'K']
target = data['label']
features = data[feature_names]

scaler.fit(features)
label_encoder.fit(target)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Perform prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])

    # Prepare the input data
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'N': [N],
        'P': [P],
        'K': [K]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction_probabilities = model.predict(input_data_scaled)
    predicted_class_index = prediction_probabilities.argmax()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

    # Render the result template with the predicted crop
    return render_template('result.html', crop=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
