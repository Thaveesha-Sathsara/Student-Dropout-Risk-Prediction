import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Load data
df = pd.read_csv('dataset.csv')
target_col = 'Target'
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define pipelines
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

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Generate feature names
def generate_feature_names(preprocessor, numerical_cols, categorical_cols):
    feature_names = []
    # Numerical names
    feature_names.extend(numerical_cols)
    # Categorical names from OneHotEncoder categories_
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    for col, categories in zip(categorical_cols, cat_encoder.categories):
        feature_names.extend([f"{col}_{cat}" for cat in categories])
    return feature_names

feature_names = generate_feature_names(preprocessor, numerical_cols, categorical_cols)

# Create dataframe
X_clean_df = pd.DataFrame(X_processed, columns=feature_names)
X_clean_df[target_col] = y.reset_index(drop=True)

# Save cleaned data
X_clean_df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved as 'cleanend_data.csv'")
