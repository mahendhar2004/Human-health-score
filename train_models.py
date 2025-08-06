
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_models():
    print("Starting model training process...")
    
    csv_filename = 'smoking_driking_dataset_Ver01.csv'
    try:
        df = pd.read_csv(csv_filename) 
    except FileNotFoundError:
        print(f"\n[ERROR] '{csv_filename}' not found.")
        print("Please make sure your dataset file is in this folder and named correctly.\n")
        return

    df['gender'] = df['sex'].replace({'Male': 1, 'Female': 0})

    features = [
        'age', 'height', 'weight', 'waistline', 'SBP', 'DBP', 'BLDS', 
        'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin', 
        'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'gender'
    ]
    
    smoking_target = 'SMK_stat_type_cd'
    drinking_target = 'DRK_YN'

    all_cols = features + [smoking_target, drinking_target]
    # Remove 'gender' from all_cols check since we create it from 'sex'
    all_cols.remove('gender') 
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] The following columns are missing from your CSV file: {missing_cols}")
        print("Please check your CSV file again.\n")
        return

    X = df[features]
    y_smoking = df[smoking_target] - 1 
    y_drinking = df[drinking_target].replace({'Y': 1, 'N': 0})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data loaded and scaled successfully.")

    print("Training smoking model...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_smoking, test_size=0.2, random_state=42)
    smoking_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    smoking_model.fit(X_train_s, y_train_s)
    print(f"Smoking model accuracy: {smoking_model.score(X_test_s, y_test_s):.4f}")

    print("Training drinking model...")
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled, y_drinking, test_size=0.2, random_state=42)
    drinking_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    drinking_model.fit(X_train_d, y_train_d)
    print(f"Drinking model accuracy: {drinking_model.score(X_test_d, y_test_d):.4f}")

    joblib.dump(smoking_model, 'smoking_model.pkl')
    joblib.dump(drinking_model, 'drinking_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\nModels and scaler have been saved successfully!")
    print("You can now run the 'app.py' backend server.")

if __name__ == '__main__':
    if os.path.basename(__file__) == 'train_models.py':
         train_and_save_models()
