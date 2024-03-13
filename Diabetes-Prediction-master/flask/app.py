import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for,session
from sklearn.preprocessing import MinMaxScaler
import pickle
from flask_sqlalchemy import SQLAlchemy
import pickle

app = Flask(__name__)
# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = 'your_secret_key_here'
# Load your machine learning model
model = pickle.load(open('C:/Users/Lenovo/Downloads/Diabetes-Prediction-master/Diabetes-Prediction-master/flask/model.pkl', 'rb'))

# Define your database model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Glucose_Level = db.Column(db.Float, nullable=False)
    Insulin = db.Column(db.Float, nullable=False)
    BMI = db.Column(db.Float, nullable=False)
    Age = db.Column(db.Float, nullable=False)
    Prediction_Result = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f"Prediction(Glucose_Level={self.Glucose_Level}, Insulin={self.Insulin}, BMI={self.BMI}, Age={self.Age}, Prediction_Result={self.Prediction_Result})"
    
model = pickle.load(open('C:/Users/Lenovo/Downloads/Diabetes-Prediction-master/Diabetes-Prediction-master/flask/model.pkl', 'rb'))
    

# Prediction route
from flask import redirect, url_for, render_template

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        Glucose_Level = float(request.form['Glucose Level'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = float(request.form['Age'])

        # Prepare features for prediction
        features = np.array([[Glucose_Level, Insulin, BMI, Age]])

        # Transform features using scaler (if any)
        features_scaled = sc.transform(features)

        # Make prediction using the model
        prediction = model.predict(features_scaled)

        # Process prediction result
        if prediction == 1:
            pred_result = "You may have Diabetes, please consult a Doctor."
            show_button = True  # Set show_button to True if the user has diabetes
        else:
            pred_result = "You may don't have Diabetes."
            show_button = False  # Set show_button to False if the user doesn't have diabetes

        # Store prediction result in the database
        prediction_record = Prediction(
            Glucose_Level=Glucose_Level,
            Insulin=Insulin,
            BMI=BMI,
            Age=Age,
            Prediction_Result=pred_result
        )
        db.session.add(prediction_record)
        db.session.commit()

        return render_template('index.html', prediction_text=pred_result, show_button=show_button)    

@app.route('/delete_prediction/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    prediction = Prediction.query.get(prediction_id)
    if prediction:
        db.session.delete(prediction)
        db.session.commit()
        return 'Prediction deleted successfully', 200
    else:
        return 'Prediction not found', 404

@app.route('/show_predictions')
def show_predictions():
    # Retrieve predictions from the database
    predictions = Prediction.query.all()
    return render_template('predictions.html', predictions=predictions)



# Load the dataset
dataset = pd.read_csv('C:/Users/Lenovo/Downloads/Diabetes-Prediction-master/Diabetes-Prediction-master/flask/diabetes.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

# Scale the dataset
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

# Define routes




@app.route('/search')
def search():
    return render_template('search.html')



# Home route
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_username = request.form['admin_username']
        admin_password = request.form['admin_password']
        
        # Check if the entered credentials match the hardcoded values
        if admin_username == 'avi' and admin_password == 'avi':
            session['admin_username'] = admin_username  # Store admin username in the session
            return redirect(url_for('show_predictions'))  # Redirect to admin dashboard
        else:
            return render_template('admin_login.html', message='Invalid admin username or password.')
    else:
        return render_template('admin_login.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            # Redirect to index.html upon successful login
            return redirect(url_for('index'))
    return render_template('login.html')

# Sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Create a new user
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        # Redirect to login page after successful sign-up
        return redirect(url_for('login'))
    return render_template('signin.html')

# Home route (index.html)
@app.route('/index')
def index():
    return render_template('index.html')

with app.app_context():
    db.create_all()
if __name__ == '__main__':
    # Create all database tables before running the app


    app.run(debug=True, port=8000)
