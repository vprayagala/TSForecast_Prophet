# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:57:46 2017

@author: vprayagala2

Driver ROutine for Web Forecast

##############################################################################
#### Revision LOG    #########################################################
# Version   Date            User            Comments
# 0.1       10/22/2017      TA ML Team      Initial Draft
##############################################################################
"""
#%%
#Set the utility path
import sys,os
#sys.path.append("C:\\git\\projects\\WebTraffic\\Util")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "util"))
#%%
#Set Logger
from time import strftime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
LOG_FILE="C:\\git\\projects\\WebTraffic\\Output\\Log\\Log_"+(strftime("%Y_%m_%d_%H_%M_%S"))+".txt"
handler = logging.FileHandler(LOG_FILE)
handler.setLevel(logging.INFO)
# add the handlers to the logger
logger.addHandler(handler)

logger.info("Start of Log\n")
#%%
#Import the packages
import argparse
import pandas as pd
#import numpy as np
import ForecastWrapper as FW
#%%
#Global Parameters
IN_FILE="C:\\git\\projects\\WebTraffic\\Data\\data-11-21.xlsx"
MODEL_DIR="C:\\git\\projects\\WebTraffic\\Output\\Model\\"
OUT_DIR="C:\\git\\projects\\WebTraffic\\Output\\Results\\"
#%%
#Import the required packages
if __name__ == '__main__':
    
    #Create commenad line argument parser
    parser=argparse.ArgumentParser()
    #Add command line argument , to control whether to train or run saved model
    parser.add_argument('--run_type',
                        help="Train or Run Saved Model")
    args=parser.parse_args()
    typ=args.run_type
    
    
    #Create Wrapper Object
    fcst_obj=FW.ForecastWrapper(logger)
    
    #Read Excel Data specify the absolute path
    data=fcst_obj.load_excel_data(IN_FILE)
    #Convert the timestamp column to timestamp data format
    data['TimeStamp']=pd.to_datetime(data['TimeStamp'])
    #Split into two data frames valid and invalid
    data_valid=data.loc[:,['TimeStamp','Valid']]
    data_invalid=data.loc[:,['TimeStamp','Invalid']]
    
    #Process data for making it prophet ready
    data_processed=fcst_obj.process_data(data_valid)
    #Plot current data
    fcst_obj.visualize_data(data_processed,OUT_DIR,"Actual")

    #Build Model if only run type is train else just get the saved model and 
    #predict
    model_file=MODEL_DIR+"WebTraffic.mdl"
    if typ == 'Train':
        #Build Model
        model=fcst_obj.build_model(data_processed,seasonality='Daily')
        #Save Model
        fcst_obj.save_model(model,model_file)
    else:
        #Read saved model
        model=fcst_obj.read_model(model_file)
    #Forecast Model
    future,forecast=fcst_obj.make_predictions(model,data_processed,period=24*4,\
                                              freq='15min')
    fcst_obj.view_results(model,forecast,OUT_DIR,"Plot")
    
    
    #Export results to json
    file_name=OUT_DIR+"Forecast_Result.json"
    with open(file_name,'w') as out_file:
        out_file.write(forecast.to_json(orient='records', lines=True,\
                                        date_format='iso'))
        