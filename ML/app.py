from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# Load your dataset
train_data_path = 'data/train.csv'  # Path to your training data
test_data_path = 'data/test.csv'    # Path to your testing data

# Load training and testing datasets
train_df = pd.read_csv(train_data_path)
print("Training Data Columns:", train_df.columns)  # Check the columns in your dataset

# Define a function to categorize the weather condition
def categorize_weather(temperature, humidity, wind_speed):
    if humidity > 70:
        return 2  # Rainy
    elif wind_speed > 10:
        return 1  # Windy
    elif temperature > 25:
        return 0  # Sunny
    else:
        return 3  # Humid

# Apply the categorization function to create a new column
train_df['weather_category'] = train_df.apply(
    lambda row: categorize_weather(row['meantemp'], row['humidity'], row['wind_speed']), axis=1)

# Prepare the features and target variable
X_train = train_df[['humidity', 'wind_speed', 'meanpressure']]  # Update based on your dataset
y_train = train_df['weather_category']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# After fitting the model
y_train_pred = model.predict(X_train_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy:.2f}")

# Calculate F1 score
f1 = f1_score(y_train, y_train_pred, average='weighted')  # You can also use 'macro' or 'micro' depending on your needs
print(f"Training F1 Score: {f1:.2f}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    pressure = float(request.form['precipitation'])

    # Prepare the input features for prediction (must match the trained model features)
    input_features = scaler.transform([[humidity, wind_speed, pressure]])  # Ensure correct order

    # Make prediction
    prediction = model.predict(input_features)[0]

    # Map the prediction to weather conditions
    weather_conditions = {
        0: 'Sunny',
        1: 'Windy',
        2: 'Rainy',
        3: 'Humid'
    }
    predicted_condition = weather_conditions.get(prediction, "Unknown")

    return render_template('index.html', prediction=predicted_condition)

if __name__ == '__main__':
    app.run(debug=True)
