"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # print(result)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    # print(type(predictions[0]))
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    # print(len(predictions))
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    accuracy = r2_score(_predictions, y_true) #accuracy_score(_predictions, y_true)
    print(accuracy)
    assert accuracy > 0.8

