import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#Dark Theme Configuration
st.set_page_config(page_title="Student Dropout Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: #f1f1f1;
        }
        .stButton > button {
            background-color: #444;
            color: white;
            border: none;
        }
        .stDataFrame > div {
            color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

#Title
st.title("🎓 Student Dropout Risk Prediction")
st.markdown("Predict dropout risk using machine learning on student data.")

#File Upload
uploaded_file = st.file_uploader("📂 Upload cleaned_data.csv", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File successfully loaded!")

    target_col = 'Target'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    #Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Evaluation
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    dropout_index = list(model.classes_).index("Dropout")
    roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')

    st.subheader("📊 Evaluation Metrics")
    st.code(classification_report(y_test, y_pred))
    st.write("🔍 **ROC AUC Score (multiclass):**", round(roc_auc, 4))

    #Dropout Risk Predictions
    full_probs = model.predict_proba(X)
    dropout_risks = full_probs[:, dropout_index]

    results_df = X.copy()
    results_df['TrueLabel'] = y.values
    results_df['DropoutRisk'] = dropout_risks

    #Display Table
    st.subheader("📋 Student Dropout Risk Table")
    st.dataframe(
        results_df[['TrueLabel', 'DropoutRisk']]
        .sort_values(by="DropoutRisk", ascending=False)
        .reset_index(drop=True),
        height=400
    )

    #Plot: Dropout Risk Distribution
    st.subheader("📉 Dropout Risk Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(results_df['DropoutRisk'], kde=True, bins=30, color='#00adb5', ax=ax)

    # Style Fixes for Dark Theme
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor("#1e1e1e")
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white', which='both')
    ax.set_title("Distribution of Predicted Dropout Risk", color='white', fontsize=14)
    ax.set_xlabel("Dropout Risk", color='white')
    ax.set_ylabel("Student Count", color='white')

    st.pyplot(fig)

    #Download Button
    st.download_button("📥 Download Results as CSV",
                       data=results_df.to_csv(index=False).encode('utf-8'),
                       file_name="student_dropout_risk_FULL.csv",
                       mime='text/csv')
else:
    st.warning("⚠️ Please upload the `cleaned_data.csv` file to begin.")
