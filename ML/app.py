from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset and preprocess
def load_and_train_model():
    # Load dataset
    file_path = 'data/DailyDelhiClimateTrain.csv'
    data = pd.read_csv(file_path)

    # Create labels for weather conditions based on feature thresholds
    conditions = []
    for i, row in data.iterrows():
        if row['humidity'] > 80:
            conditions.append(1)  # Rainy
        elif row['wind_speed'] > 5:
            conditions.append(2)  # Windy
        elif row['meantemp'] > 30:
            conditions.append(0)  # Sunny
        else:
            conditions.append(3)  # Humid

    data['condition'] = conditions

    # Features and target variable
    X = data[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
    y = data['condition']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    with open('weather_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler

# Load the model and scaler
model, scaler = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the inputs
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])  # Treat as pressure since precipitation is absent

        # Scale the inputs
        inputs = np.array([[temp, humidity, wind_speed, precipitation]])
        inputs_scaled = scaler.transform(inputs)

        # Predict the weather
        prediction = model.predict(inputs_scaled)
        
        # Map prediction to readable format
        weather_conditions = {0: "Sunny", 1: "Rainy", 2: "Windy", 3: "Humid"}  # Example mapping
        result = weather_conditions.get(prediction[0], "Unknown")  # Use .get to avoid KeyError

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
