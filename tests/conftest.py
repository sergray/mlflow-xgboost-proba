"""
Test fixtures for mlflow_xgboost_proba tests
"""

import os
import shutil
import subprocess
import tempfile

import pytest
from mlflow import MlflowClient
from xgboost import XGBClassifier

from tests.fixtures import (
    MAX_MODEL_SERVE_WAIT,
    MLFLOW_MODEL_PATH,
    MLFLOW_MODEL_PORT,
    MLFLOW_MODEL_URI,
    MLFLOW_SERVER_PORT,
    MLFLOW_TRACKING_URI,
    clean_dirs,
    stop_subprocesses,
    test_data_fixture,
)


@pytest.fixture()
def temp_model_path():
    temp_path = tempfile.mkdtemp(prefix="xgb_proba_")
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def test_data():
    return test_data_fixture()


@pytest.fixture(scope="session")
def xgb_model(test_data):
    X_train, X_test, y_train, y_test = test_data
    # create model instance
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic")
    # fit model
    bst.fit(X_train, y_train)
    return bst


@pytest.fixture(scope="session")
def xgb_proba_model(xgb_model):
    import mlflow_xgboost_proba as xgboost_proba

    if os.path.exists(MLFLOW_MODEL_PATH):
        shutil.rmtree(MLFLOW_MODEL_PATH)

    xgboost_proba.save_model(xgb_model, MLFLOW_MODEL_PATH)

    return xgboost_proba.load_model(MLFLOW_MODEL_PATH)


@pytest.fixture(scope="session")
def mlflow_server():
    # if not os.path.isdir(MLFLOW_DIR):
    #     os.mkdir(MLFLOW_DIR)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    proc = subprocess.Popen(
        [  # noqa: S603, S607
            "mlflow",
            "server",
            # "--default-artifact-root",
            # os.path.join(MLFLOW_DIR, "mlruns"),
            # "--backend-store-uri",
            # os.path.join(MLFLOW_DIR, "mlruns"),
            # "--artifacts-destination",
            # os.path.join(MLFLOW_DIR, "mlartifacts"),
            "--workers",
            "1",
            "--port",
            str(MLFLOW_SERVER_PORT),
        ]
    )
    print(f"Started mlflow server with pid: {proc.pid}")

    yield

    stop_subprocesses(proc.pid)
    print(f"Stopped mlflow server with pid: {proc.pid}")
    del os.environ["MLFLOW_TRACKING_URI"]
    # shutil.rmtree(MLFLOW_DIR)
    clean_dirs("mlruns", "mlartifacts")


@pytest.fixture()
def mlflow_model_serve(xgb_proba_model):
    proc = subprocess.Popen(
        [  # noqa: S603, S607
            "mlflow",
            "models",
            "serve",
            "--env-manager",
            "local",
            "-m",
            MLFLOW_MODEL_PATH,
            "-w",
            "1",
            "-p",
            str(MLFLOW_MODEL_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode is None
    print(f"Started mlflow model serve with pid: {proc.pid}")

    for i in range(MAX_MODEL_SERVE_WAIT):
        line = proc.stderr.readline()
        print(line)
        if f"Listening at: {MLFLOW_MODEL_URI}" in line:
            break
        print(f"{i + 1}/{MAX_MODEL_SERVE_WAIT} waiting for mlflow model serve to start...")
        import time

        time.sleep(1)

    yield

    stop_subprocesses(proc.pid)
    print(f"Stopped mlflow model serve with pid: {proc.pid}")


@pytest.fixture()
def mlflow_client():
    return MlflowClient(MLFLOW_TRACKING_URI)
