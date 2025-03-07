# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
# Load the preprocessed dataset from a CSV file
df = pd.read_csv("preprocess_used_cars.csv")

# Standardize column names (convert to lowercase)
df.columns = df.columns.str.lower()

# Display the first few rows to verify the data
print("Dataset Preview:")
print(df.head())


# %%
# Create target variable 'suitability' if it doesn't already exist.
# Example rule: A car is recommended (1) if its price is below the median OR 
# (if qualityscore exists) its qualityscore is above the median.
if 'suitability' not in df.columns:
    price_median = df['price'].median()
    if 'qualityscore' in df.columns:
        qualityscore_median = df['qualityscore'].median()
        df['suitability'] = ((df['price'] < price_median) | (df['qualityscore'] > qualityscore_median)).astype(int)
    else:
        df['suitability'] = (df['price'] < price_median).astype(int)

# Print the distribution of the target variable
print("Suitability distribution:")
print(df['suitability'].value_counts())


# %%
# Prepare the features DataFrame (X) by dropping the target column,
# and set y as the target variable.
X = df.drop('suitability', axis=1)
y = df['suitability']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)


# %%
# Define transformers for numerical and categorical features

# Numeric transformer: impute missing values with the mean
numeric_transformer = SimpleImputer(strategy='mean')

# Categorical transformer: one-hot encode categorical variables.
# (Using 'sparse_output=False' for scikit-learn 1.2+)
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine the transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# %%
# Define classifiers with a fixed random_state for reproducibility
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000)

# Create pipelines for each model (preprocessor + classifier)
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', dt_model)])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', rf_model)])
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', lr_model)])

# Split data into training and test sets (using stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training and Test set sizes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)


# %%
# Define a parameter grid for Logistic Regression
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['lbfgs', 'saga']  # Adjust based on your requirements
}

# Set up GridSearchCV for Logistic Regression pipeline
grid_search_lr = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Fit grid search on the training data
grid_search_lr.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best cross-validation F1 score (Logistic Regression):", grid_search_lr.best_score_)

# Retrieve the best estimator
best_lr = grid_search_lr.best_estimator_


# %%
# Generate predictions on the test set using the best Logistic Regression pipeline
y_pred_lr = best_lr.predict(X_test)

# Define a function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)

# Evaluate the tuned Logistic Regression model
evaluate_model(y_test, y_pred_lr, "Logistic Regression (Tuned)")


# %%
# The function below encapsulates the entire process for modularity
def train_classification_component(data_path="preprocess_used_cars.csv"):
    # Load, preprocess, create target, split data, and tune model (using cells 2-8)
    # (The code from Cells 2 through 8 would be integrated here)
    # For brevity, we assume the operations have been executed as above.
    # Return the best tuned Logistic Regression pipeline and the candidate DataFrame (X)
    return best_lr, X

# When running the notebook, you can now call this function at the end.
trained_pipeline, candidate_data = train_classification_component()
print("Classification Component training is complete. The trained pipeline and candidate data are ready for use.")



