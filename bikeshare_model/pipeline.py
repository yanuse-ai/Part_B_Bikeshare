import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import ColumnDropper
from bikeshare_model.processing.features import yr_mapping
from bikeshare_model.processing.features import mnth_mapping
from bikeshare_model.processing.features import season_mapping
from bikeshare_model.processing.features import weather_mapping
from bikeshare_model.processing.features import holiday_mapping
from bikeshare_model.processing.features import workingday_mapping
from bikeshare_model.processing.features import hour_mapping
from bikeshare_model.processing.data_manager import numerical_features
from bikeshare_model.processing.data_manager import unused_colms


bikeshare_pipe=Pipeline([
    ('weekday_imputer', WeekdayImputer('weekday')),
    ('weathersit_imputer', WeathersitImputer('weathersit')),
    ('yr_mapper', yr_mapping),
    ('mnth_mapper', mnth_mapping),
    ('season_mapper', season_mapping),
    ('weather_mapper', weather_mapping),
    ('holiday_mapper', holiday_mapping),
    ('workingday_mapper', workingday_mapping),
    ('hour_mapper', hour_mapping),
    ('outlier_handler', OutlierHandler(numerical_features)),
    ('weekday_encoder', WeekdayOneHotEncoder('weekday')),
    ('column_dropper', ColumnDropper(unused_colms)),
    ('regressor', RandomForestRegressor())
])