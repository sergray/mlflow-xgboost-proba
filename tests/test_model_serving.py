# ruff: disable=unused-import

import numpy as np
import requests

from tests.fixtures import (
    MLFLOW_MODEL_URI,
)

MODEL_ENDPOINT = f"{MLFLOW_MODEL_URI}/invocations"


def test_model_serve(test_data, mlflow_model_serve):
    X_train, X_test, y_train, y_test = test_data

    inference_request = {
        "inputs": X_test.tolist(),
    }

    response = requests.post(MODEL_ENDPOINT, json=inference_request, timeout=3)

    json_response = response.json()

    predictions = json_response["predictions"]

    assert np.array(predictions).shape == (y_test.shape[0], 3)
