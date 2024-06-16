import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB

# Load datasets
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')
submission_format = pd.read_csv('submission_format.csv')

# Merge training features and labels on respondent_id
train_data = pd.merge(train_features, train_labels, on='respondent_id')

# Separate features and targets
X_train = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_train = train_data[['xyz_vaccine', 'seasonal_vaccine']]
X_test = test_features.drop(columns=['respondent_id'])

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Ensure there are no NaNs in the preprocessed data
if np.any(np.isnan(X_train_preprocessed)):
    print("NaNs found in X_train_preprocessed")
if np.any(np.isnan(X_test_preprocessed)):
    print("NaNs found in X_test_preprocessed")

# Split training data for validation
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=42)

# Define models to be evaluated
models = {
    'LogisticRegression': MultiOutputClassifier(LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)),
    'RandomForest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    'GradientBoosting': MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)),
    'SVM': MultiOutputClassifier(SVC(probability=True, random_state=42)),
    'BernoulliNB': MultiOutputClassifier(BernoulliNB()),
    'GaussianNB': MultiOutputClassifier(GaussianNB())
}

# Function to train and evaluate models
def evaluate_models(models, X_train_split, y_train_split, X_valid, y_valid):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train_split, y_train_split)
        
        y_pred_proba = model.predict_proba(X_valid)
        y_pred_proba_xyz = y_pred_proba[0][:, 1]
        y_pred_proba_seasonal = y_pred_proba[1][:, 1]
        
        roc_auc_xyz = roc_auc_score(y_valid['xyz_vaccine'], y_pred_proba_xyz)
        roc_auc_seasonal = roc_auc_score(y_valid['seasonal_vaccine'], y_pred_proba_seasonal)
        mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2
        
        results[model_name] = {
            'roc_auc_xyz': roc_auc_xyz,
            'roc_auc_seasonal': roc_auc_seasonal,
            'mean_roc_auc': mean_roc_auc
        }
        
        print(f"{model_name} - ROC AUC (xyz_vaccine): {roc_auc_xyz:.4f}, ROC AUC (seasonal_vaccine): {roc_auc_seasonal:.4f}")
        print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
    
    return results

# Evaluate models
model_results = evaluate_models(models, X_train_split, y_train_split, X_valid, y_valid)

# Select the best model based on the highest mean ROC AUC
best_model_name = max(model_results, key=lambda k: model_results[k]['mean_roc_auc'])
best_model = models[best_model_name]

# Hyperparameter tuning for GradientBoosting and LogisticRegression (Optional)
if best_model_name == 'GradientBoosting' or best_model_name == 'LogisticRegression':
    param_grid = {}
    
    if best_model_name == 'GradientBoosting':
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__learning_rate': [0.01, 0.1],
            'estimator__max_depth': [3, 5, 7]
        }
    elif best_model_name == 'LogisticRegression':
        param_grid = {
            'estimator__C': [0.01, 0.1, 1, 10]
        }

    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_preprocessed, y_train)

    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_valid)
    y_pred_proba_xyz = y_pred_proba[0][:, 1]
    y_pred_proba_seasonal = y_pred_proba[1][:, 1]

    # Calculate ROC AUC for both targets
    roc_auc_xyz = roc_auc_score(y_valid['xyz_vaccine'], y_pred_proba_xyz)
    roc_auc_seasonal = roc_auc_score(y_valid['seasonal_vaccine'], y_pred_proba_seasonal)

    print(f"Best {best_model_name} - ROC AUC (xyz_vaccine): {roc_auc_xyz:.4f}, ROC AUC (seasonal_vaccine): {roc_auc_seasonal:.4f}")
    print(f"Mean ROC AUC: {(roc_auc_xyz + roc_auc_seasonal) / 2:.4f}")

# Train the best model on the full training set
best_model.fit(X_train_preprocessed, y_train)

# Predict probabilities on the test set
y_pred_proba = best_model.predict_proba(X_test_preprocessed)
y_pred_proba_xyz = y_pred_proba[0][:, 1]
y_pred_proba_seasonal = y_pred_proba[1][:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': y_pred_proba_xyz,
    'seasonal_vaccine': y_pred_proba_seasonal
})

submission.to_csv('submission.csv', index=False)
