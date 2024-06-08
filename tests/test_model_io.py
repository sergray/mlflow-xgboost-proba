# ruff: disable=unused-import

import os

import yaml

import mlflow_xgboost_proba as xgboost_proba


def test_save_model(xgb_model, temp_model_path):
    xgboost_proba.save_model(xgb_model, temp_model_path)

    with open(os.path.join(temp_model_path, "MLmodel")) as manifest_file:
        model_manifest = yaml.safe_load(manifest_file)

    with open(os.path.join(temp_model_path, "requirements.txt")) as requirements_file:
        requirements = requirements_file.read()

    assert model_manifest["flavors"]["python_function"]["loader_module"] == "mlflow_xgboost_proba"
    assert "mlflow_xgboost_proba==" in requirements

    mlflow_model = xgboost_proba.load_model(temp_model_path)

    assert mlflow_model


def test_predict(test_data, xgb_proba_model):
    X_train, X_test, y_train, y_test = test_data
    y_predicted = xgb_proba_model.predict(X_test)
    assert y_predicted.shape == y_test.shape


def test_predict_proba(test_data, xgb_proba_model):
    X_train, X_test, y_train, y_test = test_data
    y_predicted = xgb_proba_model.predict_proba(X_test)
    assert y_predicted.shape == (y_test.shape[0], 3)


def test_predict_with_probabilities(test_data, xgb_proba_model):
    X_train, X_test, y_train, y_test = test_data
    y_predicted = xgb_proba_model.predict(X_test, {"with_proba": True})
    assert y_predicted.shape == (y_test.shape[0], 3)
