<!-- # AI-Model--Fido

Project Overview
Data Extraction: Data is sourced from MongoDB collections (users, securitykeys, keyassignments), loaded into DataFrames, and merged to form a unified dataset.

Data Preprocessing:
Missing values are handled (e.g., filling lastUsed).
Date columns are converted to datetime objects.
A new feature, days_since_assignment, calculates the days since a key was assigned.

Feature Engineering:
Target variable: key_assigned is set to 1 if a key is assigned, otherwise 0.
Categorical variables (role, department, status_user) are label-encoded.

Model Training and Evaluation:
Data is split into training and testing sets.
A Random Forest Classifier (RFC) is trained and tested, achieving an accuracy of 100% on the test set.

Why Random Forest Classifier?
Random Forest is a versatile and efficient model, particularly suitable for this binary classification problem due to:
Its ability to handle both categorical and numerical features.
Robustness to overfitting, making it a reliable choice for high-accuracy predictions.

Model Evaluation
The model’s performance is measured with:

Accuracy: 100%
Classification Report: All metrics show a perfect score.
Confusion Matrix: Indicates no misclassifications

Files
Model Artifacts: evaluation_metrics.pkl (accuracy, classification report, and confusion matrix).
Label Encoders: le_role.pkl, le_department.pkl, le_status.pkl.

How to Run
Clone the repository and ensure MongoDB access, then run the script to train and evaluate the model.





 -->
