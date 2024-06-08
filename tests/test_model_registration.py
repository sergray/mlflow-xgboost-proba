# ruff: disable=unused-import

import os
import tempfile

import mlflow
import yaml

import mlflow_xgboost_proba


def test_log_model(mlflow_server, mlflow_client, xgb_proba_model, test_data):
    mlflow_model_name = ""
    X_train, X_test, y_train, y_test = test_data
    predicted_data = xgb_proba_model.predict_proba(X_test)
    signature = mlflow.models.infer_signature(X_test, predicted_data)
    run_name = f"test_log_model_{mlflow_model_name}"

    assert len(mlflow.search_runs()) == 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_xgboost_proba.log_model(xgb_proba_model, artifact_path="model", signature=signature)

    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'")

    assert len(runs) == 1

    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_client.download_artifacts(
            run_id=run.info.run_id,
            path="model",
            dst_path=temp_dir,
        )
        with open(os.path.join(temp_dir, "model", "MLmodel")) as manifest_file:
            model_manifest = yaml.safe_load(manifest_file)

        with open(os.path.join(temp_dir, "model", "requirements.txt")) as requirements_file:
            requirements = requirements_file.read()

    assert model_manifest["flavors"]["python_function"]["loader_module"] == "mlflow_xgboost_proba"
    assert "mlflow_xgboost_proba==" in requirements
