import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np
import os

def main():
    # Load Table 1 (brain volumes)
    table1 = pd.read_csv('volumes_GT.csv')  # Replace with the actual path to Table 1

    # Load Table 2 (ground truth labels)
    table2 = pd.read_csv('GT.csv')  # Replace with the actual path to Table 2

    # Remove "Segmentation_" prefix from scan names in Table 1
    table1['Scan'] = table1['Scan'].str.replace('Final_', '', regex=False)
    table1['Scan'] = table1['Scan'].str.replace('.nii', '', regex=False)

    # Normalize scan names in Table 2
    table2['Scan'] = table2['Scan'].str.replace('.nii.gz', '', regex=False)

    # Merge the two tables on the 'Scan' column using outer join
    merged_data = pd.merge(table1, table2, on='Scan', how='outer')

    # Drop rows with missing ground truth labels
    cleaned_data = merged_data.dropna(subset=['GT'])
    cleaned_data = cleaned_data.dropna(subset=['Vent', 'Sub', 'White'])

    # Calculate total brain volume
    cleaned_data['Total_Brain_Volume'] = (
        cleaned_data['Vent'].fillna(0) +
        cleaned_data['Sub'].fillna(0) +
        cleaned_data['White'].fillna(0)
    )

    # Define features and target variable
    X = cleaned_data[['Vent', 'Sub', 'White', 'Total_Brain_Volume']]
    y = cleaned_data['GT']
    scan_names = cleaned_data['Scan']  # Keep track of scan names

    # Convert categorical target variable to numerical
    y = y.map({'Norm': 0, 'NPH': 1})

    # Compute class weights to address class imbalance
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Initialize the RBF SVM classifier with class weights
    clf = make_pipeline(
        StandardScaler(), 
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight=class_weight_dict)
    )

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, scan_names_train, scan_names_test = train_test_split(
        X, y, scan_names, test_size=0.4, random_state=42
    )

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_proba = clf.predict_proba(X_test)

    # Print prediction probabilities for debugging
    for scan, proba in zip(scan_names_test, y_proba):
        print(f'Scan: {scan}, Probabilities: {proba}')

    # Adjust threshold to reduce false positives for NPH
    threshold = 0.5  # Example threshold, adjust as needed
    y_pred = (y_proba[:, 1] > threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Norm', 'NPH'])

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    # Print predictions for each test scan
    for scan, pred in zip(scan_names_test, y_pred):
        print(f'Scan: {scan}, Prediction: {"NPH" if pred == 1 else "Norm"}')

    # Train on the full dataset and save the model (optional)
    clf.fit(X, y)
    joblib.dump(clf, os.path.join('./nph_classifiers', 'rbf_svm_GT.pkl'))
    print('Classifier saved to rbf_svm_GT.pkl')

if __name__ == "__main__":
    main()
