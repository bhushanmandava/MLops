import wandb  # Weights and Biases for experiment tracking
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Initialize a new WandB run
wandb.init(project="first-project-mlops-course", name="data-model-tracking")

# Step 2: Load the original dataset
iris = datasets.load_iris()
X_original = iris.data
y = iris.target 

# Step 3: Log the original dataset to WandB
artifact_original = wandb.Artifact(name="original_dataset", type="dataset")
artifact_original.add(wandb.Table(data=X_original, columns=iris.feature_names), name="original_data")
wandb.log_artifact(artifact_original)

# Step 4: Preprocess the dataset (Cleaned Version)
scaler = StandardScaler()
X_cleaned = scaler.fit_transform(X_original)

artifact_cleaned = wandb.Artifact(name="cleaned_dataset", type="dataset")
artifact_cleaned.add(wandb.Table(data=X_cleaned, columns=iris.feature_names), name="cleaned_data")
wandb.log_artifact(artifact_cleaned)

# Step 5: Feature Engineering (Feature-Engineered Version)
X_feature_engineered = np.c_[X_cleaned, X_cleaned[:, 0] * X_cleaned[:, 1]]
feature_names = iris.feature_names + ["feature_0_1_product"]

artifact_feature_engineered = wandb.Artifact(name="feature_engineered_dataset", type="dataset")
artifact_feature_engineered.add(wandb.Table(data=X_feature_engineered, columns=feature_names), name="feature_engineered_data")
wandb.log_artifact(artifact_feature_engineered)

# Step 6: Train and Version Multiple Models
hyperparameters = {
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42
}
wandb.config.update(hyperparameters)

# Train and log model on original dataset
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)
model_original = RandomForestClassifier(**hyperparameters)
model_original.fit(X_train_orig, y_train)
accuracy_orig = accuracy_score(y_test, model_original.predict(X_test_orig))

wandb.log({"accuracy_original": accuracy_orig})

model_filename_orig = "rf_model_original.pkl"
joblib.dump(model_original, model_filename_orig)

artifact_original_model = wandb.Artifact(name="original_model", type="model")
artifact_original_model.add_file(model_filename_orig)
wandb.log_artifact(artifact_original_model)

# Train and log model on cleaned dataset
X_train_cleaned, X_test_cleaned, _, _ = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)
model_cleaned = RandomForestClassifier(**hyperparameters)
model_cleaned.fit(X_train_cleaned, y_train)
accuracy_cleaned = accuracy_score(y_test, model_cleaned.predict(X_test_cleaned))

wandb.log({"accuracy_cleaned": accuracy_cleaned})

model_filename_cleaned = "rf_model_cleaned.pkl"
joblib.dump(model_cleaned, model_filename_cleaned)

artifact_model_cleaned = wandb.Artifact(name="cleaned_model", type="model")
artifact_model_cleaned.add_file(model_filename_cleaned)
wandb.log_artifact(artifact_model_cleaned)

# Train and log model on feature-engineered dataset
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_feature_engineered, y, test_size=0.2, random_state=42)
model_feature_engineered = RandomForestClassifier(**hyperparameters)
model_feature_engineered.fit(X_train_fe, y_train_fe)
accuracy_feature_engineered = accuracy_score(y_test_fe, model_feature_engineered.predict(X_test_fe))

wandb.log({"accuracy_feature_engineered": accuracy_feature_engineered})

model_feature_engineered_filename = "rf_model_feature_engineered.pkl"
joblib.dump(model_feature_engineered, model_feature_engineered_filename)

artifact_model_feature_engineered = wandb.Artifact(name="feature_engineered_model", type="model")
artifact_model_feature_engineered.add_file(model_feature_engineered_filename)
wandb.log_artifact(artifact_model_feature_engineered)

# Step 7: Finish the WandB run
wandb.finish()
