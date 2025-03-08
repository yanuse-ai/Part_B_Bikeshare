
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer, Mapper



def test_weather_variable_imputer(sample_input_data):
    # print(sample_input_data[0].head())
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config_.weathersit_var,  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[12230,'weathersit']) 

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[12230,'weathersit'] == "Clear"


def test_season_mapper(sample_input_data):
    # print(sample_input_data[0].head())
    # Given
    transformer = Mapper(
        variables='season',
        mappings ={'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
    )
    # assert np.isnan(sample_input_data[0].loc[12830,'season']) 

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[12830,'season'] == 2