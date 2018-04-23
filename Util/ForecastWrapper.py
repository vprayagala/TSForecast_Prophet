# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:57:46 2017

@author: vprayagala2

Utility Class for Prophet - More methods to be added

##############################################################################
#### Revision LOG    #########################################################
# Version   Date            User            Comments
# 0.1       10/22/2017      TA ML Team      Initial Draft
##############################################################################
"""
#%%
#Load the required Libraries
#from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import pickle
#%%
#Get the tran data
class ForecastWrapper:
    def __init__(self,logger):
        #self.logger=logging.getLogger(__name__)
        self.logger=logger
        
    def load_excel_data(self,file_name):
        data= pd.read_excel(file_name)
        self.logger.info("Data Dimensions (rows,columns)")
        self.logger.info(data.shape)
        self.logger.info("Column Data Types")
        self.logger.info(data.dtypes)
        return data
        
    def process_data(self,data,freq='min'):
        data['ds']=data.iloc[:,0]
        data['y']=data.iloc[:,1]
        data.loc[data['y'] == 0,'y'] = np.NaN
        self.logger.info(data.head())
        columns_to_drop=[column for column in data.columns if column not in ['ds','y']]
        data.drop(columns_to_drop,inplace=True,axis=1)
        
        #Create Index with date range with minutes intervel
        #min_date=min(data["ds"])
        #max_date=max(data["ds"])
        #logging.info("Min Date = {} and Max Date = {}".format(min_date,max_date))
        #index = pd.date_range(start=min_date,end=max_date,freq=freq)
        #data.set_index(data['ds'],inplace=True)
        #Re-index data frame
        #data=data.reindex(index=index,fill_value=0)
    
        #data.drop(['ds'],axis=1,inplace=True)
        #data.reset_index(inplace=True)
        data.columns=['ds','y']
        data['cap']=max(data['y'])
        data['floor']=0
        self.logger.info(data.head())
        return data
    
    def visualize_data(self,data,out_dir,fig_no):
        #Visualize the Data
        fig_name=out_dir+"Data_"+fig_no+".png"
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.plot(data['ds'],data['y'])
        fig.savefig(fig_name)   # save the figure to file
        plt.close(fig)
        
    def build_model(self,data,seasonality):
        model=Prophet(growth='logistic',interval_width=0.95,\
                      changepoint_prior_scale=0.7)
        if seasonality not in ['Yearly','Monthly','Quarterly','Weekly','Daily','Hourly']:
            print("Invalid Seasonality Parameter:{}".format(seasonality))
            self.logger.error("Invalid Seasonality Parameter:{}".format(seasonality))
            return None
        else:
            if seasonality == 'Yearly':
                model.add_seasonality(name='Yearly',period=365,fourier_order=7)
            if seasonality == 'Monthly':
                model.add_seasonality(name='Monthly',period=30,fourier_order=7)  
            if seasonality == 'Quarerly':
                model.add_seasonality(name='Quarerly',period=120,fourier_order=7)
            if seasonality == 'Weekly':
                model.add_seasonality(name='Weekly',period=7,fourier_order=7)
            if seasonality == 'Daily':
                model.add_seasonality(name='Daily',period=1,fourier_order=7)
            if seasonality == 'Hourly':
                model.add_seasonality(name='Hourly',period=int(1/24),fourier_order=7)
            model.fit(data)
            return model
    
    def build_model_with_holidays(self,data,seasonality,holidays):
        model=Prophet(growth='logistic',\
                      interval_width=0.95,\
                      n_changepoints=8,\
                      changepoint_prior_scale=0.7,\
                      holidays=holidays)
        model.add_seasonality(name='hourly',period=8,fourier_order=7)
        model.fit(data)
        return model        
        
    def make_predictions(self,model,data,period=15,freq='min'):
        future=model.make_future_dataframe(periods=period,freq=freq)
        future['cap']=max(data['y'])
        future['floor']=0
        forecast=model.predict(future)
        return future,forecast
    
    def save_model(self,model,model_file):
        with open(model_file,"wb") as f:
            pickle.dump(model,f)
            
    def read_model(self,model_file):
        with open(model_file,"rb") as f:
            model=pickle.load(f)            
            return model
        
    def view_results(self,model,forecast,out_dir,fig_no):
        fig_name=out_dir+"Forecast_"+fig_no+".png"
        fig=model.plot(forecast)
        fig.savefig(fig_name)
        plt.close(fig)
        
    def add_holidays_seasonality(self,data):
        start_date=min(data["ds"]).date()
        end_date=max(data["ds"]).date()
        delta = timedelta(days=1)
        d = start_date
        diff = 0
        weekend = set([5, 6])
        week_end=[]
        
        while d <= end_date:
            if d.weekday() in weekend:
                diff += 1
                week_end.append(d)
            d += delta
        
        holidays = pd.DataFrame({
          'holiday': 'weekends',
          'ds': list(pd.to_datetime(week_end)),
          'lower_window': 0,
          'upper_window': 1,
        })
        return holidays

    def add_offhours_seasonality(self,data):
        start_date=min(data["ds"]).date()
        end_date=max(data["ds"]).date()
        delta = timedelta(days=1)
        d = start_date
        diff = 0
        weekend = set([5, 6])
        week_end=[]
        
        while d <= end_date:
            if d.weekday() not in weekend:
                diff = 17*60
                for i in range(14*60):
                    x=pd.to_datetime(d)+timedelta(minutes=diff)
                    diff+=1
                    week_end.append(x)
            d += delta
        
        offdays = pd.DataFrame({
          'holiday': 'offhours',
          'ds': list(pd.to_datetime(week_end)),
          'lower_window': 0,
          'upper_window': 0,
        })
        return offdays      
