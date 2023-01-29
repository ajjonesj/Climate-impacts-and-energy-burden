# -*- coding: utf-8 -*-
"""
This script uses downloaded LOCA csv files (point-estimates) across 31 models
and from 2014 - 2099. The outputs are csv files separated by RCPs (4.5 and 8.5) and 
the historical and future temperatures for all runs, models, and for the 10 most relevant models. 

@author: Andrew Jones
"""


import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------
path =  'Insert the location of the project file'
# Download data
LOCA_historical = pd.read_csv(path + 
    '/Data/LOCA_Pre_Processing/Average_Daily_Temperature_by_runs.csv').set_index(
    "Unnamed: 0")
LOCA_historical.index = pd.to_datetime(LOCA_historical.index)
LOCA_future = pd.read_csv(path + 
    '/Data/LOCA_Pre_Processing/Phoenix/Future/2020-2099/Projections/Average_Daily_Temperature_by_runs.csv').set_index(
    "Unnamed: 0")
LOCA_future.index = pd.to_datetime(LOCA_future.index)
      
#Separates dataframes to RCP 4.5 and 8.5
LOCA_historical_RCP45 = LOCA_historical.filter(regex='rcp45')
LOCA_historical_RCP85 = LOCA_historical.filter(regex='.rcp85')
LOCA_future_RCP45 = LOCA_future.filter(regex='.rcp45')
LOCA_future_RCP85 = LOCA_future.filter(regex='.rcp85')

#Removes the unnecssary portion of the lables
LOCA_historical_RCP45.columns = LOCA_historical_RCP45.columns.str.rstrip('.1.rcp45')
LOCA_historical_RCP85.columns = LOCA_historical_RCP85.columns.str.rstrip('.1.rcp85')
LOCA_future_RCP45.columns = LOCA_future_RCP45.columns.str.rstrip('.1.rcp45')
LOCA_future_RCP85.columns = LOCA_future_RCP85.columns.str.rstrip('.1.rcp85')

#Saves
LOCA_future_RCP45.to_excel(path + '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/All Runs/LOCA_future_RCP45.xlsx')
LOCA_future_RCP85.to_excel(path + '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/All Runs/LOCA_future_RCP85.xlsx')
LOCA_historical_RCP45.to_excel(path + 
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/All Runs/LOCA_historical_RCP45.xlsx')
LOCA_historical_RCP85.to_excel(path +
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/All Runs/LOCA_historical_RCP85.xlsx')


def extract_model_avgs(LOCA_historical_RCP45, LOCA_historical_RCP85):
    "Separates the scenarios by their models"
    
    #Create an empty dataframe 
    LOCA_historical_RCP45_models = pd.DataFrame()
    LOCA_historical_RCP85_models = pd.DataFrame()
    
    for cols, model in enumerate(LOCA_historical_RCP85.columns.str.split('-').str[0].unique()):
        #Agencies with multiple model runs are averages and grouped into their corresponding models 
        rcp45_agencies = np.mean(LOCA_historical_RCP45.filter(regex=model), axis=1)
        rcp85_agencies = np.mean(LOCA_historical_RCP85.filter(regex=model), axis=1)
        
        #Inserts the agencies into their respective model slots
        LOCA_historical_RCP45_models.insert(cols, model, rcp45_agencies)
        LOCA_historical_RCP85_models.insert(cols, model, rcp85_agencies)
    return (LOCA_historical_RCP45_models, LOCA_historical_RCP85_models)


[LOCA_historical_RCP45_models,
 LOCA_historical_RCP85_models] = extract_model_avgs(LOCA_historical_RCP45,
                                                    LOCA_historical_RCP85)
                                                    
[LOCA_future_RCP45_models,
 LOCA_future_RCP85_models] = extract_model_avgs(LOCA_future_RCP45,
                                                LOCA_future_RCP85)

LOCA_historical_RCP45_models.to_excel(path+
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Models/LOCA_historical_RCP45_models.xlsx')
LOCA_historical_RCP85_models.to_excel(path +
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Models/LOCA_historical_RCP85_models.xlsx')
LOCA_future_RCP45_models.to_excel(path +
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Models/LOCA_future_RCP45_models.xlsx')
LOCA_future_RCP85_models.to_excel(path +
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Models/LOCA_future_RCP85_models.xlsx')

#Extracts the top 10 models of interests
models_of_focus = [LOCA_historical_RCP45.columns[0], LOCA_historical_RCP45.columns[4],
                   LOCA_historical_RCP45.columns[5], LOCA_historical_RCP45.columns[6],
                   LOCA_historical_RCP45.columns[30], LOCA_historical_RCP45.columns[9],
                   LOCA_historical_RCP45.columns[13], LOCA_historical_RCP45.columns[18],
                   LOCA_historical_RCP45.columns[19], LOCA_historical_RCP45.columns[25]]

LOCA_historical_RCP45_PX_runs = LOCA_historical_RCP45.loc[:, LOCA_historical_RCP45.columns.isin(models_of_focus)]
LOCA_historical_RCP85_PX_runs = LOCA_historical_RCP85.loc[:, LOCA_historical_RCP85.columns.isin(models_of_focus)]
LOCA_future_RCP45_PX_runs = LOCA_future_RCP45.loc[:, LOCA_future_RCP45.columns.isin(models_of_focus)]
LOCA_future_RCP85_PX_runs = LOCA_future_RCP85.loc[:, LOCA_future_RCP85.columns.isin(models_of_focus)]

LOCA_historical_RCP45_PX_runs.to_excel(path +
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Phoenix 10 Runs/LOCA_historical_RCP45_PX_runs.xlsx')
LOCA_historical_RCP85_PX_runs.to_excel(path+
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Phoenix 10 Runs/LOCA_historical_RCP85_PX_runs.xlsx')
LOCA_future_RCP45_PX_runs.to_excel(path+
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Phoenix 10 Runs/LOCA_future_RCP45_PX_runs.xlsx')
LOCA_future_RCP85_PX_runs.to_excel(path+
    '/Data/LOCA_Pre_Processing/Phoenix/Cleaned Data/Phoenix 10 Runs/LOCA_future_RCP85_PX_runs.xlsx')
