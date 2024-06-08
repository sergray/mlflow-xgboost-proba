import os
import shutil

import psutil
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MLFLOW_MODEL_PATH = "./tests/data/mlflow_model"
MLFLOW_SERVER_PORT = 5555
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_SERVER_PORT}"
MLFLOW_MODEL_PORT = 5500
MLFLOW_MODEL_URI = f"http://127.0.0.1:{MLFLOW_MODEL_PORT}"

MAX_MODEL_SERVE_WAIT = 5


def test_data_fixture():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data["data"], data["target"], test_size=0.2)
    return (X_train, X_test, y_train, y_test)


def stop_subprocesses(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


def clean_dirs(*dirs):
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
