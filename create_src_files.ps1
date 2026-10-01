New-Item -ItemType Directory -Force -Path "src" | Out-Null

Set-Content -Path "src\validate_data.py" -Value @'
import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_telco_schema():
    """Defines the strict schema specification for the Telco Churn dataset."""
    return pa.DataFrameSchema({
        "customerID": Column(pa.String),
        "gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(pa.Int, Check.isin([0, 1])),
        "Partner": Column(pa.String, Check.isin(["Yes", "No"])),
        "Dependents": Column(pa.String, Check.isin(["Yes", "No"])),
        "tenure": Column(pa.Int, Check.ge(0)),
        "PhoneService": Column(pa.String, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(pa.String, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(pa.String, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "OnlineBackup": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "DeviceProtection": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "TechSupport": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(pa.String, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(pa.String, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(pa.String, Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])),
        "MonthlyCharges": Column(pa.Float, Check.ge(0.0)),
        "TotalCharges": Column(pa.String, nullable=True),
        "Churn": Column(pa.String, Check.isin(["Yes", "No"]))
    }, strict=True)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_telco_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False

if __name__ == "__main__":
    data_path = 'data/raw/churn.csv'
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
'@

Set-Content -Path "src\preprocess_pipeline.py" -Value @'
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    data_path = 'data/raw/churn.csv'
    df = pd.read_csv(data_path)

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))

    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")

if __name__ == "__main__":
    run_preprocessing()
'@

Set-Content -Path "src\validate_outputs.py" -Value @'
import os
import json
import numpy as np

def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []

    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")

    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape)
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum())
        },
        "errors": errors
    }

    os.makedirs("artifacts", exist_ok=True)
    report_path = 'artifacts/preprocessing_summary_report.json'

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[SUCCESS] Output Validation PASSED.")
        print("[INFO] Features are clean, scaled, encoded, and dimensionally consistent.")
        print(f"[INFO] Summary report saved to {report_path}")
        return True

if __name__ == "__main__":
    validate_preprocessing_outputs()
'@

Set-Content -Path "src\train_registry.py" -Value @'
import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    mlflow.set_experiment("Telco_Churn_Prediction")

    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        return

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "class_weight": "balanced"
    }

    with mlflow.start_run(run_name="RandomForest_Registry_V2") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_metrics(metrics)

        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name="Telco_Churn_Production_Model"
        )

        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: 'Telco_Churn_Production_Model'")

if __name__ == "__main__":
    train_and_register_model()
'@

Set-Content -Path "src\automate_lifecycle.py" -Value @'
import mlflow
from mlflow.tracking import MlflowClient

def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"
    metric_to_optimize = "recall"

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")

        new_models = [mv for mv in all_versions if mv.current_stage == "None"]
        if new_models:
            print(f"\n[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")
            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"  -> Version {mv.version} is now in Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not staging_models:
            print("\n[INFO] No models in Staging to evaluate. Exiting.")
            return

        print("\n[INFO] Evaluating Staging candidates...")
        best_challenger = None
        best_challenger_score = -1.0

        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"  -> Staging Candidate: Version {mv.version} | {metric_to_optimize}: {score:.4f}")

            if score > best_challenger_score:
                best_challenger_score = score
                best_challenger = mv

        print(f"\n[INFO] Best Challenger is Version {best_challenger.version} ({metric_to_optimize}: {best_challenger_score:.4f})")

        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None
        promote_challenger = False

        if not current_champion:
            print("[INFO] No model currently in Production. Challenger wins by default!")
            promote_challenger = True
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = champ_run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Current Production Champion: Version {current_champion.version} | {metric_to_optimize}: {champ_score:.4f}")

            if best_challenger_score > champ_score:
                print("[SUCCESS] Challenger beat the Champion!")
                promote_challenger = True
            else:
                print("[INFO] Challenger failed to beat the Champion. Champion retains its title.")

        if promote_challenger:
            print(f"\n[INFO] Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            if current_champion:
                print(f"[INFO] Archiving defeated Champion (Version {current_champion.version})...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )

        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]

        if remaining_staging:
            print("\n[INFO] Cleaning up Staging area...")
            for mv in remaining_staging:
                print(f"  -> Archiving Version {mv.version} (Lost the competition)")
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
                )

        print("\n[SUCCESS] Automated Model Lifecycle execution complete!")

    except Exception as e:
        print(f"[ERROR] Failed to automate lifecycle: {e}")

if __name__ == "__main__":
    automate_champion_challenger()
'@

Set-Content -Path "src\generate_registry_report.py" -Value @'
import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"

    try:
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)

        if not champion:
            print("[ERROR] No model found in Production stage!")
            return

        run = client.get_run(champion.run_id)

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision")
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl"
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")

if __name__ == "__main__":
    generate_registry_report()
'@

Write-Host "Done. Created 6 files in src:"
Get-ChildItem "src" -Name