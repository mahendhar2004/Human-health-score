
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

smoking_model = None
drinking_model = None
scaler = None
population_df = None

try:
    smoking_model = joblib.load('smoking_model.pkl')
    drinking_model = joblib.load('drinking_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Models and scaler loaded successfully.")

    csv_filename = 'smoking_driking_dataset_Ver01.csv'
    population_df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
    population_df['gender'] = population_df['sex'].replace({'Male': 1, 'Female': 0})
    print(f"'{'smoking_driking_dataset_Ver01.csv'}' loaded for population analysis.")

except FileNotFoundError:
    print("\n[ERROR] Critical files not found. Please run train_models.py first.\n")
    
def get_health_suggestions(data, smoking_is_risk, drinking_is_risk):
    """Generates a list of personalized health suggestions based on input data and model predictions."""
    suggestions = []
    
    if data.get('LDL_chole', 0) > 130:
        suggestions.append({
            "topic": "LDL Cholesterol",
            "recommendation": f"Your LDL (bad) cholesterol is high at {data.get('LDL_chole')}. Reducing saturated fats (found in red meat and full-fat dairy) and increasing soluble fiber (oats, apples, beans) can help lower it."
        })
    if data.get('HDL_chole', 0) < 40:
        suggestions.append({
            "topic": "HDL Cholesterol",
            "recommendation": f"Your HDL (good) cholesterol is low at {data.get('HDL_chole')}. Regular aerobic exercise and including healthy fats like those in avocados and nuts can help raise it."
        })
    if data.get('SGOT_AST', 0) > 40 or data.get('SGOT_ALT', 0) > 55:
        recommendation_text = f"Your liver enzymes AST ({data.get('SGOT_AST')}) and/or ALT ({data.get('SGOT_ALT')}) are elevated, suggesting liver inflammation. "
        if drinking_is_risk:
            recommendation_text += "This is strongly linked to your 'High Risk' drinking prediction. Reducing alcohol intake is critical."
        else:
            recommendation_text += "Consult a doctor to investigate potential causes, which can include diet, medications, or other conditions."
        suggestions.append({"topic": "Liver Enzymes (AST/ALT)", "recommendation": recommendation_text})
    if data.get('urine_protein', 0) > 1:
         suggestions.append({
            "topic": "Urine Protein",
            "recommendation": f"The presence of protein in your urine (level {data.get('urine_protein')}) can be an early sign of kidney issues. It is highly recommended to consult a doctor to monitor your kidney function."
        })

    try:
        height_m = data.get('height', 0) / 100
        weight_kg = data.get('weight', 0)
        if height_m > 0:
            bmi = weight_kg / (height_m ** 2)
            if bmi >= 25:
                suggestions.append({"topic": "Body Mass Index (BMI)", "recommendation": f"Your BMI of {bmi:.1f} is in the overweight range. This can contribute to higher blood pressure and blood sugar. Focusing on a balanced diet and regular physical activity can significantly improve these metrics."})
    except (TypeError, ZeroDivisionError):
        pass

    if data.get('gamma_GTP', 0) > 55:
        recommendation_text = f"Your Gamma-GTP level ({data.get('gamma_GTP')}) is elevated, another strong indicator of liver stress. "
        if drinking_is_risk:
            recommendation_text += "This reinforces the need to address alcohol consumption."
        suggestions.append({"topic": "Liver Enzyme (Gamma-GTP)", "recommendation": recommendation_text})

    if data.get('SBP', 0) > 130 or data.get('DBP', 0) > 85:
        suggestions.append({"topic": "Blood Pressure", "recommendation": f"Your blood pressure of {data.get('SBP')}/{data.get('DBP')} is high. To manage this, consider reducing sodium intake and practicing regular cardiovascular exercise."})
    
    if data.get('BLDS', 0) > 100:
        suggestions.append({"topic": "Blood Sugar", "recommendation": f"Your fasting blood sugar of {data.get('BLDS')} is in the pre-diabetic range. Focus on a diet rich in fiber and limit sugary drinks and refined carbs."})
            
    if data.get('triglyceride', 0) > 150:
        recommendation_text = f"High triglycerides ({data.get('triglyceride')}) increase cardiovascular risk. "
        if drinking_is_risk:
            recommendation_text += "This is often linked to alcohol and sugar intake, so addressing your drinking habits is a key first step."
        else:
            recommendation_text += "This can be improved by reducing sugar and refined carbohydrate intake."
        suggestions.append({"topic": "Triglycerides", "recommendation": recommendation_text})
            
    if len(suggestions) == 0:
        suggestions.append({"topic": "Overall Health", "recommendation": "Excellent work! Your key health metrics are within optimal ranges. Maintain your healthy habits to continue this positive trend."})
            
    return suggestions

@app.route('/predict', methods=['POST'])
def predict():
    if not smoking_model:
        return jsonify({'error': 'Models not loaded. Please check backend server console.'}), 500

    data = request.get_json()

    feature_names = [
        'age', 'height', 'weight', 'waistline', 'SBP', 'DBP', 'BLDS', 
        'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin', 
        'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'gender'
    ]
    
    backend_data = {
        'age': data.get('age'), 'height': data.get('height(cm)'), 'weight': data.get('weight(kg)'),
        'waistline': data.get('waist(cm)'), 'SBP': data.get('systolic'), 'DBP': data.get('relaxation'),
        'BLDS': data.get('fasting blood sugar'), 'hemoglobin': data.get('hemoglobin'),
        'serum_creatinine': data.get('serum creatinine'), 'gamma_GTP': data.get('gamma-GTP'),
        'triglyceride': data.get('triglyceride'), 'gender': data.get('gender'),
        'HDL_chole': data.get('HDL_chole'), 'LDL_chole': data.get('LDL_chole'),
        'urine_protein': data.get('urine_protein'), 'SGOT_AST': data.get('SGOT_AST'),
        'SGOT_ALT': data.get('SGOT_ALT'),
        'tot_chole': data.get('HDL_chole', 0) + data.get('LDL_chole', 0) + (0.2 * data.get('triglyceride', 0))
    }

    input_df = pd.DataFrame([backend_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    smoking_prediction_proba = smoking_model.predict_proba(input_scaled)[0]
    drinking_prediction_proba = drinking_model.predict_proba(input_scaled)[0]

    smoking_risk_prob = smoking_prediction_proba[2] if len(smoking_prediction_proba) > 2 else 0
    drinking_risk_prob = drinking_prediction_proba[1] if len(drinking_prediction_proba) > 1 else 0

    smoking_is_risk = bool(smoking_risk_prob > 0.4)
    drinking_is_risk = bool(drinking_risk_prob > 0.5)

    suggestions = get_health_suggestions(backend_data, smoking_is_risk, drinking_is_risk)
    
    response = {
        'smokingPrediction': {
            'isRisk': smoking_is_risk,
            'probability': float(round(smoking_risk_prob * 100, 1)) 
        },
        'drinkingPrediction': {
            'isRisk': drinking_is_risk,
            'probability': float(round(drinking_risk_prob * 100, 1))
        },
        'insights': suggestions
    }

    return jsonify(response)

@app.route('/get_population_data', methods=['GET'])
def get_population_data():
    if population_df is None:
        return jsonify({'error': 'Population dataset not loaded.'}), 500

    sample_df = population_df.sample(n=2000, random_state=42)
    
    overall_scores = []
    age_vs_organ_score = []

    for _, row in sample_df.iterrows():
        health_score = 100
        organ_score_sum = 0
        organ_score_count = 0

        if row['SBP'] > 120 or row['DBP'] > 80: health_score -= 15
        if row['BLDS'] > 100: health_score -= 10
        if row['gamma_GTP'] > 55: health_score -= 12
        if row['triglyceride'] > 150: health_score -= 10
        if row['LDL_chole'] > 130: health_score -= 10
        
        organ_metrics = {'SBP': row['SBP'], 'DBP': row['DBP'], 'BLDS': row['BLDS'], 
                         'serum_creatinine': row['serum_creatinine'], 'gamma_GTP': row['gamma_GTP'],
                         'SGOT_AST': row['SGOT_AST'], 'SGOT_ALT': row['SGOT_ALT']}
        
        for key, value in organ_metrics.items():
            organ_score_sum += 100 if pd.isna(value) else 50 # Simplified scoring for population
            organ_score_count += 1

        overall_scores.append(max(0, health_score))
        if organ_score_count > 0:
            age_vs_organ_score.append({'x': row['age'], 'y': organ_score_sum / organ_score_count})

    return jsonify({
        'overall_scores': overall_scores,
        'age_vs_organ_score': age_vs_organ_score
    })

if __name__ == '__main__':
    if os.path.basename(__file__) == 'app.py':
        app.run(debug=True)
