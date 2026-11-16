from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Model aur Preprocessor load karein
model_path = 'model/Best_Random_Forest_Model.pkl'
preprocessor_path = 'model/preprocessor.pkl'

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.route('/')
def home():
    return render_template('index.html', form_data = {})

@app.route('/predict', methods=['POST'])
def predict():
    
        try:
            # 1. Form se user ka input data collect karein
            gender = request.form['gender']
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            smoking_history = request.form['smoking_history']
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = int(request.form['blood_glucose_level'])

            # 2. DataFrame banayein (exact column names ke sath jo training mein the)
            input_data = pd.DataFrame([[
                gender, age, hypertension, heart_disease, 
                smoking_history, bmi, HbA1c_level, blood_glucose_level
            ]], columns=[
                'gender', 'age', 'hypertension', 'heart_disease', 
                'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
            ])

            # 3. Preprocessor ke through data transform karein (Scaling & One-Hot Encoding)
            processed_data = preprocessor.transform(input_data)

            # 4. Model se prediction karwayein
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)[0][1] # Diabetes hone ki probability

            # 5. Result message set karein
            if prediction[0] == 1:
                result_text = f"Result: Positive for Diabetes (Risk Probability: {probability * 100:.2f}%)"
            else:
                result_text = f"Result: Negative for Diabetes (Safe, Probability: {(1 - probability) * 100:.2f}%)"

            return render_template('index.html', prediction_text=result_text, form_data = request.form)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8000)