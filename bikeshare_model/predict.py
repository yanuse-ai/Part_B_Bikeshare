import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import pre_pipeline_preparation
from bikeshare_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bikeshare_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bikeshare_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bikeshare_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    # data_in={
    #         'dteday':['2012-11-05'], 
    #         'season':['winter'],
    #         'hr':['6am'],
    #         'holiday':['No'],
    #         'weekday':['Mon'],
    #         'workingday':['Yes'],
    #         'weathersit':['Mist'],
    #         'temp':[6.10],	
    #         'atemp':[3.0014],
    #         'hum':[49.0],
    #         'windspeed':[19.0012],
    #         'casual':[4],
    #         'registered':[134],
    #         'cnt': [0]
    #         }
    #             'cnt':[139]

    # 2012-02-19,spring,1pm,No,Sun,No,Clear,6.1,3.998000000000001,49.0,11.0014,64,197,261
    # 2012-02-19,spring,1pm,No,Sun,No,Clear,6.1,3.998000000000001,49.0,11.0014,64,197,261
    data_in={
            'dteday':["2012-02-19"], 
            'season':['spring'],
            'hr':['1pm'],
            'holiday':['No'],
            'weekday':['Sun'],
            'workingday':['No'],
            'weathersit':['Clear'],
            'temp':[6.1],	
            'atemp':[3.9980],
            'hum':[49.0],
            'windspeed':[11.0014],
            'casual':[64],
            'registered':[197],
            'cnt': [0]
            }
    make_prediction(input_data=data_in)