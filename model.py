import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")
target_col = 'Target'

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split (only for evaluation — NOT for prediction)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate on test set
y_pred = model.predict(X_test)
y_prob_all = model.predict_proba(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob_all, multi_class='ovr')
print("ROC AUC Score (multiclass):", roc_auc)

#Predict on the FULL dataset
all_probs = model.predict_proba(X)

# Get the index of 'Dropout' class
dropout_index = list(model.classes_).index('Dropout')

# Get dropout probabilities for each student
dropout_risks = all_probs[:, dropout_index]

# Add dropout probability and true label to the original dataset
results_df = X.copy()
results_df['TrueLabel'] = y.values
results_df['DropoutRisk'] = dropout_risks

# Preview first 10 students
print("\nSample of predicted dropout risk for all students:")
print(results_df[['TrueLabel', 'DropoutRisk']].head(10))

# Save the full prediction result
results_df.to_csv("student_dropout_risk_FULL.csv", index=False)
print("\n Saved FULL dropout predictions for all students to 'student_dropout_risk_FULL.csv'")
