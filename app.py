import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def clean_data(df, target_col='Target'):
    if target_col not in df.columns:
        raise ValueError("Target column not found!")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].reset_index(drop=True)

    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # Apply
    X_processed = preprocessor.fit_transform(X)

    # Get final column names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    feature_names = numerical_cols.copy()
    for col, cats in zip(categorical_cols, cat_encoder.categories_):
        feature_names.extend([f"{col}_{cat}" for cat in cats])

    # Combine final DataFrame
    cleaned_df = pd.DataFrame(X_processed, columns=feature_names)
    cleaned_df[target_col] = y

    return cleaned_df


#STREAMLIT UI
st.set_page_config(page_title="ML Dataset Cleaner", layout="wide")
st.title("📊 Machine Learning Data Cleaner")

uploaded_file = st.file_uploader("Upload raw CSV dataset with a 'Target' column", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📥 Raw Uploaded Data")
        st.dataframe(df.head(10))

        if st.button("🚿 Clean My Data"):
            cleaned_df = clean_data(df)
            st.success("✅ Data cleaned successfully!")

            st.subheader("🧼 Cleaned Data Preview")
            st.dataframe(cleaned_df.head(10))

            # Download button
            csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Cleaned Data", csv_data, "cleaned_data.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
