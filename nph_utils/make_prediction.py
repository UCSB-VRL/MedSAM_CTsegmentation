import os
import numpy as np
import pandas as pd
import csv
import joblib

def make_prediction():
    BASE = ''  # Update with the correct base path if needed
    print('------------ making prediction -------------')
    
    # Load classifier
    classifier_name = 'rbf_svm_GT.pkl'
    vol_name = 'volumes_GT.csv'
    classifier_path = '/data/home/umang/Vader_umang/Seg_models/MedSAM/nph_classifiers/rbf_svm_GT.pkl'
    vol_path = os.path.join('/data/home/umang/Vader_umang/Seg_models/MedSAM/files_store', vol_name)
    predictions_csv = os.path.join('/data/home/umang/Vader_umang/Seg_models/MedSAM/files_store', 'predictions_GT.csv')
    
    # Load the classifier
    with open(classifier_path, 'rb') as f:
        clf = joblib.load(f)
    
    # Load and process ratio data from CSV file
    dfvol = pd.read_csv(vol_path)
    
    # Sort by 'Scan' column to ensure consistent ordering
    dfvol_sorted = dfvol.sort_values(by='Scan')
    
    # Open the file for writing predictions
    with open(predictions_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scan', 'Prediction'])
        
        # Extract scaler from the pipeline
        scaler = clf.named_steps['standardscaler']
        
        for _, corresp_row_ratio in dfvol_sorted.iterrows():
            prediction = 'no NPH'
            patient = corresp_row_ratio['Scan']
            vent = corresp_row_ratio['Vent']
            sub = corresp_row_ratio['Sub']
            white = corresp_row_ratio['White']
            
            # Prepare input for prediction
            x = np.array([[vent, sub, white, vent + sub + white]])
            
            # Scale the data using the same scaler used during training
            #x_scaled = scaler.transform(x)
            
            # Make prediction
            y_proba = clf.predict_proba(x)
            
            # Adjust threshold to reduce false positives for NPH
            threshold = 0.8  # Example threshold, adjust as needed
            y_pred = (y_proba[:, 1] > threshold).astype(int)
            if y_pred == 1:
                prediction = 'possible NPH'
            
            # Print and write prediction
            print(f'{patient}: {y_proba}, {prediction}')
            writer.writerow([patient, prediction])
    
if __name__ == "__main__":
    make_prediction()
