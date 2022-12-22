# -*- coding: utf-8 -*-
"""
This script calculates the new consumption with changes to Betas

@author: Andrew Jones
"""

import pandas as pd
import numpy as np
from functions_and_labels import labels,filtered_hh, add_demo_data_ele_sim,\
    add_race_income
import matplotlib.pyplot as plt
import seaborn as sns
import string
import datetime as dt

path = 'C:/Users/andre/Box/Andrew Jones_PhD Research/Climate change impacts on future electricity consumption and energy burden'
[income_group_axis_labels,income_group_numbers,income_group_labels,
 income_group_legend_labels,year_for_graphs] = labels()

#%% Loads data
CDD_slope_model = pd.read_pickle(path + '/CDD_slopes_models.pkl')
annual_cooling_points = pd.read_csv(path + '/Data/Baseline/Individual_Cooling_Balance_points.csv').set_index("BILACCT_K")
annual_heating_points = pd.read_csv(path + '/Data/Baseline/Individual_Heating_Balance_points.csv').set_index("BILACCT_K")
intercept_coef = pd.read_excel(path + '/Data/Baseline/2015_2019_intercept_coefficients.xlsx',header=0,
                               index_col=(0))
CDD_coeff = pd.read_excel(path + '/Data/Baseline/2015_2019_CDD_coefficients.xlsx',
                         header=0,
                          index_col=(0))

exogen= pd.read_csv(path+'/Data/exogen_data_mod.csv', parse_dates=[0],
                       index_col=[0])

survey_all = pd.read_stata('C:/Users/andre/Box/RAPID COVID Residential Energy/Arizona_Full datasets/Arizona 2019-2020 dataset/RET_2017_part1.dta')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7,4,5,6,2,1,8,3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"]!=""].unique()):
    survey_all.loc[survey_all["VINCOME"]==ig_num,["VINCOME_new"]] = int(ig_unsorted[idxs])
survey = survey_all[survey_all["BILACCT_K"].isin(annual_cooling_points.index)].set_index("BILACCT_K")

#Making a dummy variable(assumes we either know that they have a programmable thermostat or don't)
survey_all.replace(
    {'Heat pump (same system heats and cools using electricity onl':'Central-Heat pump',
     'AC unit packaged with gas heating (sometimes called a gas pa': 'Central-Gas',
    'Separate AC system that only cools': 'Central-Separate AC',
    "Don't know":"Central-Unknown",
    "65-74 yrs old":"65 yrs or older", 
    "75+ yrs old":"65 yrs or older"}, inplace= True)

summary_stats_AC = pd.concat([survey_all.groupby("VINCOME").describe()["VACSER1"],
survey_all.groupby("VHH_AGECODE").describe()["VACSER1"],
survey_all.groupby("VETHNIC").describe()["VACSER1"]],axis=0)
survey_all.VACSER1.median()

share_of_days_cooling = pd.read_excel(path + '/Data/Baseline/household_share.xlsx').set_index("BILACCT_K")
year_labels =  ["15/16", "16/17", "17/18", "18/19"]
X = survey_all[survey_all.BILACCT_K.isin(filtered_hh(CDD_coeff).index)]

X.set_index("BILACCT_K", inplace=True)
X.loc[:,"Share"] = filtered_hh(share_of_days_cooling)["17/18"]
X_models = X.loc[:,["VACSER1",'VFANNUM','VHOUSEHOLD','VACTYPE',
                    'VACUNITS', "Share",
                    'VBANDWELL', 'VBANSQFEET',"VRESAGE"]]
X_models["Intercept"] = 1  
X_models = pd.concat([X_models,
                    #Cooling Infrastructure
                    pd.get_dummies(X_models["VACUNITS"][X_models["VACUNITS"]!= ""],
                                   prefix= "AC_units"),
                    pd.get_dummies(X_models["VACTYPE"][(
                        (X_models["VACTYPE"]!="")& 
                        (X_models["VACTYPE"]!="Central-Gas"))]),
                    #Housing Infrastructure  
                    pd.get_dummies(X_models["VBANSQFEET"][
                          X_models["VBANSQFEET"] !=""], prefix= "Sqft"),
                     pd.get_dummies(X_models["VBANDWELL"][
                          X_models["VBANDWELL"] != ""])
                    ],axis=1)

X_models.drop(axis=1, labels=["AC_units_One",
                                "Central-Heat pump",
                            "Single family home",
                            "Sqft_Less than 1,500"], inplace= True)

marker_styles = ['+','s', '.', 'v','*', 'h','x','d' ]
[income_group_axis_labels,income_group_numbers,income_group_labels,
 income_group_legend_labels,year_for_graphs] = labels()
#LOCA data 
LOCA_historical_RCP45_PX_runs =pd.read_excel(path +'/Data/CMIP5_10_Models/LOCA_historical_RCP45_PX_runs.xlsx',
                         header=0,parse_dates=[0],
                          index_col=(0))
LOCA_historical_RCP85_PX_runs = pd.read_excel(path +'/Data/CMIP5_10_Models/LOCA_historical_RCP85_PX_runs.xlsx',
                         header=0,parse_dates=[0],
                          index_col=(0))
LOCA_future_RCP45_PX_runs = pd.read_excel(path +'/Data/CMIP5_10_Models/LOCA_future_RCP45_PX_runs.xlsx',
                         header=0,parse_dates=[0],
                          index_col=(0))
LOCA_future_RCP85_PX_runs = pd.read_excel(path +'/Data/CMIP5_10_Models/LOCA_future_RCP85_PX_runs.xlsx',
                         header=0,parse_dates=[0],
                          index_col=(0))
climate_years_range = pd.DataFrame(data=[
    pd.date_range(start= "05-01-2020",  periods = 50,
                  freq=pd.DateOffset(years=1)),
    pd.date_range(start= "04-30-2021",  periods = 50,
                  freq=pd.DateOffset(years=1))],
    ).T
climate_years_range.columns = ["start_date", "end_date"]
climate_years_range=climate_years_range.astype(str)
climate_years_range_dec = climate_years_range.copy(deep= True)
climate_years_range_dec["Decade"] = np.floor(np.arange(2020,2070)/10)*10
climate_years_range_dec["Labels"] = climate_years_range.loc[
    :,"start_date"].str.split("-").str[0]+"/" +\
    climate_years_range.loc[:,"end_date"].str.split("-").str[0] 
#Removes the households with no shares because it means they do not have a 2017-2018 model
X_variables= X_models[~(X_models.Share.isnull())]
X_variables = X_variables.loc[:,['Intercept', 'Share', 'VFANNUM', 'VACSER1', 'AC_units_3 or more',
       'AC_units_Two', 'Central-Separate AC', 'Central-Unknown', 'VHOUSEHOLD',
       'VRESAGE', 'Sqft_1,500 - 2,999', 'Sqft_3,000 or more',
       'Apartment/Condo/Townhouse', 'Mobile home']] 

#%% Calculates the new betas
def climate_projections_new_Betas():
    """Outputs the new betas and saves cooling sha
    Model inputs: 
    RCP_45_input    
    RCP_85_input
    model_input_year: This input identifies the trained fixed effect model that the climodels_2014_2019.loc[mate-driven electricity consumption are derived from 
    climate_model_input_year: This input is the climate model of interest 
        """
    #Average temperatures within the time frame
    avg_daily_temp = exogen["temp_avg"].groupby(exogen.index.date).mean()
    avg_daily_temp.index = pd.to_datetime(avg_daily_temp.index)
    avg_daily_temp = avg_daily_temp.loc[start_date_model:end_date_model]
    
    #Create a df for each household's balance point for the year of focus
    degree_days_cp = pd.concat(
        [annual_cooling_points[model_input_year], 
         annual_heating_points[model_input_year]
                                ], axis=1).dropna(axis=0)
    degree_days_cp.columns = ["CBP", "HBP"]
    
    #Create df for RCP tempeature projections for the days within the year of focus
    rcp45_in_range = LOCA_future_RCP45_PX_runs.loc[climate_years_range.loc[0,"start_date"]:
                                      climate_years_range.loc[49,"end_date"],:]
    rcp85_in_range = LOCA_future_RCP85_PX_runs.loc[climate_years_range.loc[0,"start_date"]:
                                     climate_years_range.loc[49,"end_date"],:]     
    
    #Creates a new df with the year's labels (i.e.,20XX/20XX)
    rcp45_in_range_dec = rcp45_in_range.copy(deep = True)
    rcp85_in_range_dec = rcp85_in_range.copy(deep = True)
    for j in range(len(climate_years_range)):
        start_date= climate_years_range.loc[j,"start_date"]
        end_date =  climate_years_range.loc[j,"end_date"]
        rcp45_in_range_dec.loc[start_date:end_date,"Label"] = climate_years_range_dec.loc[j,"Labels"]
        rcp85_in_range_dec.loc[start_date:end_date,"Label"] = climate_years_range_dec.loc[j,"Labels"]
    hhs_annual_VCDD_45= []
    hhs_annual_VCDD_85= []
    hhs_cooling_share_45= []
    hhs_cooling_share_85= []
    hhs_monthly_VCDD_45 = []
    hhs_monthly_VCDD_85 = []
    for idx,house in enumerate(degree_days_cp[~
            (degree_days_cp.CBP.isnull())].index.values):
        hh_CBP = degree_days_cp.loc[house,"CBP"]
        hh_HBP = degree_days_cp.loc[house,"HBP"]
        
        hh_VHDD_45 = (hh_HBP-rcp45_in_range).clip(lower=0)
        hh_VHDD_85 = (hh_HBP-rcp85_in_range).clip(lower=0)
            
        hh_VCDD_45 = (rcp45_in_range.subtract(hh_CBP)).clip(lower=0)
        hh_VCDD_85 = (rcp85_in_range.subtract(hh_CBP)).clip(lower=0)
        
        hh_VHDD_45.loc[:,"Labels"] = rcp45_in_range_dec.loc[:,"Label"]
        hh_VHDD_85.loc[:,"Labels"] = rcp85_in_range_dec.loc[:,"Label"]
        hh_VCDD_45.loc[:,"Labels"] = rcp45_in_range_dec.loc[:,"Label"]
        hh_VCDD_85.loc[:,"Labels"] = rcp85_in_range_dec.loc[:,"Label"]
        
        annual_hh_VCDD_45 = hh_VCDD_45.groupby("Labels").sum()
        annual_hh_VCDD_85 = hh_VCDD_85.groupby("Labels").sum()
        
        monthly_hh_VCDD_45 = hh_VCDD_45.groupby(["Labels",
                                                 hh_VCDD_45.index.month]).sum()
        monthly_hh_VCDD_85 = hh_VCDD_85.groupby(["Labels",
                                                 hh_VCDD_85.index.month]).sum()
        #------------------------Cooling Shares-----------------------
        #Sets temperatures greater than CBP to 1 and less than or equal to to 0
        #while keeping the df structure
        hh_df_share_45 = rcp45_in_range_dec.mask(
            rcp45_in_range_dec.iloc[:,:-1]>hh_CBP,other = 1).mask(
                rcp45_in_range_dec.iloc[:,:-1]<=hh_CBP,other = 0)
        hh_df_share_85 = rcp85_in_range_dec.mask(
            rcp85_in_range_dec.iloc[:,:-1]>hh_CBP,other = 1).mask(
                rcp85_in_range_dec.iloc[:,:-1]<=hh_CBP,other = 0)
        #Relabel the years
        hh_df_share_45.loc[:,"Labels"] = rcp45_in_range_dec.loc[:,"Label"]  
        hh_df_share_85.loc[:,"Labels"] = rcp85_in_range_dec.loc[:,"Label"]  
        
        #Calculates the share of the days in a year greater than CBP
        hh_cooling_share_45 = hh_df_share_45.groupby("Labels").sum()/\
            hh_df_share_45.groupby("Labels").count()
        hh_cooling_share_85 = hh_df_share_85.groupby("Labels").sum()/\
            hh_df_share_85.groupby("Labels").count()
        annual_hh_VCDD_45.loc[:,"acct"] = house
        annual_hh_VCDD_85.loc[:,"acct"] = house
        monthly_hh_VCDD_45.loc[:,"acct"] = house
        monthly_hh_VCDD_85.loc[:,"acct"] = house
        hh_cooling_share_45.loc[:,"acct"] = house
        hh_cooling_share_85.loc[:,"acct"] = house
        hhs_annual_VCDD_45.append(annual_hh_VCDD_45)  
        hhs_annual_VCDD_85.append(annual_hh_VCDD_85)
        hhs_monthly_VCDD_45.append(monthly_hh_VCDD_45)  
        hhs_monthly_VCDD_85.append(monthly_hh_VCDD_85)
        hhs_cooling_share_45.append(hh_cooling_share_45)
        hhs_cooling_share_85.append(hh_cooling_share_85)
    
    #Creates a dataframe for all of the households 
    hhs_annual_VCDD_45_all = pd.concat(hhs_annual_VCDD_45,
                                       axis=0).reset_index().melt(["acct","Labels"],
                                                      var_name = "models", 
                                                      value_name = "VCDD")
    
    hhs_annual_VCDD_85_all = pd.concat(hhs_annual_VCDD_85,
                                       axis=0).reset_index().melt(["acct","Labels"],
                                                      var_name = "models", 
                                                      value_name = "VCDD")
    hhs_monthly_VCDD_45_all = pd.concat(hhs_monthly_VCDD_45,
                                        axis=0).reset_index().melt(["acct","Labels",
                                        "Unnamed: 0"],
                                        var_name = "models", 
                                      value_name = "VCDD")
    hhs_monthly_VCDD_45_all.rename(columns= {"Unnamed: 0":"Months"},
                                   inplace= True)
    hhs_monthly_VCDD_85_all = pd.concat(hhs_monthly_VCDD_85,
                                       axis=0).reset_index().melt(["acct","Labels",
                                                                   "Unnamed: 0"],
                                      var_name = "models", 
                                      value_name = "VCDD")
    hhs_monthly_VCDD_85_all.rename(columns= {"Unnamed: 0":"Months"},
                                  inplace= True) 
    hhs_cooling_share_45_all = pd.concat(
        hhs_cooling_share_45,axis=0).reset_index().drop("Label",
                                                        axis=1).melt(
                                                            ["acct","Labels"],
                                                      var_name = "models", 
                                                      value_name = "VCDD")
    hhs_cooling_share_85_all = pd.concat(
        hhs_cooling_share_85,axis=0).reset_index().drop("Label",
                                                        axis=1).melt(["acct","Labels"],
                                                      var_name = "models", 
                                                      value_name = "VCDD")

    hhs_annual_VCDD_45_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP45.csv")
    hhs_annual_VCDD_85_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP85.csv")
    
    hhs_monthly_VCDD_45_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP45_monthly.csv")
    hhs_monthly_VCDD_85_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP85_monthly.csv")

    hhs_cooling_share_45_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_RCP45.csv")
    hhs_cooling_share_85_all.to_csv(path+"/Data/New_Beta_Estimates/All_hhs_RCP85.csv")

    new_betas_hhs_all_45 = []
    for idx1,yr_labels in enumerate(hhs_cooling_share_45_all.Labels.unique()): #loops through the years
        for idx2,mod_name in enumerate(hhs_cooling_share_45_all.models.unique()): #loops through the models
            #Copies a new X variables df to make sure unintentional overwriting is not occuring
            X_variables_45 = X_variables.copy(deep=True)
            #Replaces the share of days CDD days based on the year and model from the saved household shares df
            X_variables_45.loc[:,"Share"] = hhs_cooling_share_45_all[(
                (hhs_cooling_share_45_all.models ==mod_name) & 
                (hhs_cooling_share_45_all.Labels ==yr_labels)& 
                (hhs_cooling_share_45_all.acct.isin(X_variables_45.index))
                )]["VCDD"].values
            for seer_new in range(1,40):# Loops through the SEER ratings 
                #Replaces the old SEER ratings with new ones
                X_variables_45.loc[(
                    (X_variables_45.VACSER1<seer_new) | 
                    (X_variables_45.VACSER1== np.nan)|
                    (X_variables_45.VACSER1.isnull())),"VACSER1"] = seer_new
                #Saves the new Betas with RCP 4.5 into a new df and drops the accounts without slopes
                Betas_df_hh = pd.DataFrame(
                    CDD_slope_model.predict(X_variables_45).dropna(),
                    columns = ["New_Betas"])
                #New columns for the model names 
                Betas_df_hh.loc[:,"models"] = mod_name
                #New columns for the Years
                Betas_df_hh.loc[:,"Year_labels"] = yr_labels
                #New columns for the SEER ratings
                Betas_df_hh.loc[:,"SEER_labels"] = str(seer_new)
                Betas_df_hh = Betas_df_hh.reset_index()
                new_betas_hhs_all_45.append(Betas_df_hh)
    # new_betas_hhs_all_45_df = new_betas_hhs_all_df
    new_betas_hhs_all_45_df = pd.concat(new_betas_hhs_all_45,axis=0)
    
    new_betas_hhs_all_85 = []
    for idx1,yr_labels in enumerate(hhs_cooling_share_85_all.Labels.unique()):#Loops through years
        for idx2,mod_name in enumerate(hhs_cooling_share_85_all.models.unique()):# Loops through the models
            #Copies the X df to ensure there is no unintentional overwriting
            X_variables_85 = X_variables.copy(deep=True)
            #Replaces the 2017-2018 share of days with future annual share based on the model 
            X_variables_85.loc[:,"Share"] = hhs_cooling_share_85_all[(
                (hhs_cooling_share_85_all.models ==mod_name) & 
                (hhs_cooling_share_85_all.Labels ==yr_labels)& 
                (hhs_cooling_share_85_all.acct.isin(X_variables_85.index))
                )]["VCDD"].values
            
            for seer_new in range(1,40):# Loops through the SEER ratings
                #Replaces the SEER rating with new SEER ratings
                X_variables_85.loc[(
                    (X_variables_85.VACSER1<seer_new) | 
                    (X_variables_85.VACSER1== np.nan)|
                    (X_variables_85.VACSER1.isnull())),"VACSER1"] = seer_new
                
                #Creates a new df for the new RCP 8.5 betas 
                Betas_df_hh = pd.DataFrame(
                    CDD_slope_model.predict(X_variables_85).dropna(),
                    columns = ["New_Betas"])
                #Inserts a new column for the GCM names
                Betas_df_hh.loc[:,"models"] = mod_name
                #Inserts a new column for the Year labels
                Betas_df_hh.loc[:,"Year_labels"] = yr_labels
                #Inserts a new columnd for the SEER ratings
                Betas_df_hh.loc[:,"SEER_labels"] = str(seer_new)
                
                Betas_df_hh = Betas_df_hh.reset_index()
                new_betas_hhs_all_85.append(Betas_df_hh)
    
    new_betas_hhs_all_85_df = pd.concat(new_betas_hhs_all_85,axis=0)
    
    new_betas_hhs_all_45_df.to_csv(path+"/Data/New_Beta_Estimates/new_betas_All_hhs_45.csv")
    new_betas_hhs_all_85_df.to_csv(path+"/Data/New_Beta_Estimates/new_betas_All_hhs_85.csv")

    return(new_betas_hhs_all_45_df,new_betas_hhs_all_85_df,
           hhs_monthly_VCDD_45_all,hhs_monthly_VCDD_85_all,
           hhs_cooling_share_45_all,hhs_cooling_share_85_all)

start_date_model= "05-01-2017"
end_date_model = "04-30-2018"
model_input_year = year_labels[2]

[new_betas_hhs_all_45_df,new_betas_hhs_all_85_df,
       hhs_monthly_VCDD_45_all,hhs_monthly_VCDD_85_all,
       hhs_cooling_share_45_all,hhs_cooling_share_85_all] = climate_projections_new_Betas()

#%% Calculates the new loads and cost
hhs_monthly_VCDD_45_all= pd.read_csv(path+"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP45_monthly.csv")
hhs_monthly_VCDD_85_all= pd.read_csv(path +"/Data/New_Beta_Estimates/All_hhs_Cooling_RCP85_monthly.csv")
new_betas_hhs_all_45_df= pd.read_csv(path+"/Data/New_Beta_Estimates/new_betas_All_hhs_45.csv")
new_betas_hhs_all_85_df= pd.read_csv(path+"/Data/New_Beta_Estimates/new_betas_All_hhs_85.csv")
summertime_monthly_future = pd.read_csv(path +'/Data/Future_Projections/summertime_consumption_estimates.csv')
#Only focuses on the households with short-run estimates
hhs_monthly_VCDD_45_all = hhs_monthly_VCDD_45_all[hhs_monthly_VCDD_45_all.acct.isin(
    summertime_monthly_future.acct.unique())]
hhs_monthly_VCDD_85_all = hhs_monthly_VCDD_85_all[hhs_monthly_VCDD_85_all.acct.isin(
    summertime_monthly_future.acct.unique())]
new_betas_hhs_all_45_df = new_betas_hhs_all_45_df[new_betas_hhs_all_45_df.BILACCT_K.isin(
    summertime_monthly_future.acct.unique())]
new_betas_hhs_all_85_df = new_betas_hhs_all_85_df[new_betas_hhs_all_85_df.BILACCT_K.isin(
    summertime_monthly_future.acct.unique())]

historical_data = pd.read_csv(path + '/Data/Consumption_Simulations/historical.csv', 
                               date_parser=('date_s'),low_memory=False)
past_data_annual= historical_data.groupby(["acct", "model"]).sum()
historical_data.date_s = historical_data.date_s.astype('datetime64[ns]')
VCDD_model = pd.read_pickle(path + '/CDD_slopes_models.pkl')
SEER_val_mean = VCDD_model.params["VACSER1"]
[SEER_val_UB,SEER_val_LB] = VCDD_model.conf_int().loc["VACSER1",:]
#-------------------------------------------------------------------------------
#Past Summers changes
#-------------------------------------------------------------------------------
summertime_monthly_past = historical_data.groupby(
     ["acct", "model",historical_data.date_s.dt.month]).sum()

summertime_monthly_past.rename_axis(index ={"date_s":"Month"},
                                    inplace=True)
summertime_monthly_past.reset_index(inplace= True)
summertime_monthly_past.loc[((summertime_monthly_past["Month"]>= 5) & 
                               (summertime_monthly_past["Month"]<= 9)),
                            "Summer"] = "Yes"
summertime_monthly_past.loc[~((summertime_monthly_past["Month"]>= 5) & 
                               (summertime_monthly_past["Month"]<= 9)),
                            "Summer"] = "No"
summertime_monthly_past= summertime_monthly_past[
    summertime_monthly_past.Summer=="Yes"].groupby(
    ["acct", "model"]).sum()
summertime_monthly_past.drop(axis=1, labels= ["Month",
                                              "Unnamed: 0"],
                             inplace= True)
avg_daily_elec = exogen["elec_cost"].groupby(exogen.index.date).mean()
avg_daily_elec_acct = exogen[["elec_cost",
                              "BILACCT_K"]].groupby(exogen.index.date).mean()

avg_daily_elec.index = avg_daily_elec.index.astype('datetime64[ns]')
annual_avg_price = avg_daily_elec.loc['2017-05-01':'2018-04-30'].mean()
annual_min_price = avg_daily_elec.loc['2017-05-01':'2018-04-30'].min()
annual_max_price = avg_daily_elec.loc['2017-05-01':'2018-04-30'].max()
summer_avg_price = avg_daily_elec.loc['2017-05-01':'2017-09-30'].mean()
summer_min_price = avg_daily_elec.loc['2017-05-01':'2017-09-30'].min()
summer_max_price = avg_daily_elec.loc['2017-05-01':'2017-09-30'].max()

#%% TODO: filter out only the households that have short-run estimates
def new_beta_estimates():
    #Combines the new slopes (i.e, RCP 45 and RCP85) into a single df
    new_betas_hhs_all_df_test = new_betas_hhs_all_85_df.copy(deep= True).dropna()
    new_betas_hhs_all_df_test.rename(columns = {"New_Betas":"RCP_85_new_betas"}, inplace= True)
    new_betas_hhs_all_df_test.loc[:,"RCP_45_new_betas"] = new_betas_hhs_all_45_df["New_Betas"]
    
    #Resets the index to be able to join df later                                                  
    new_betas_hhs_all_df_test.set_index(["Year_labels","models","BILACCT_K",  
                                         "SEER_labels"],
                                        inplace = True)
    #Combines the new slopes (i.e, RCP 45 and RCP85) into a single df with old slopes (2017-2018)
    hhs_CDD_old_slopes= CDD_coeff.copy(deep=True)
    
    hhs_CDD_slopes = new_betas_hhs_all_df_test.join(hhs_CDD_old_slopes["17/18"].dropna(),
                                                    on ="BILACCT_K")
    hhs_CDD_slopes.rename(columns = {"New_Betas": "RCP_45_new_betas",
                                     "17/18": "Original_Betas"}, inplace= True)
    hhs_CDD_slopes.index.rename({"BILACCT_K": "acct"}, inplace= True)
    
    #Combines the VCDD into a single df for the monthly values
    hhs_monthly_VCDD_all_summer = hhs_monthly_VCDD_45_all.copy(deep=True)
    hhs_monthly_VCDD_all_summer = hhs_monthly_VCDD_all_summer.join(
        hhs_monthly_VCDD_85_all["VCDD"],rsuffix= "_RCP85")
    hhs_monthly_VCDD_all_summer.rename(columns = {"VCDD":"VCDD_RCP45"}, inplace= True)
    
    #Only keep the accounts that are in the CDD slopes df
    hhs_monthly_VCDD_all_summer = hhs_monthly_VCDD_all_summer[
    hhs_monthly_VCDD_all_summer.acct.isin(
        hhs_CDD_slopes.index.get_level_values(
        "acct"))]
    
    #Summing the VCDD for the summer based on accounts, model and year 
    hhs_monthly_VCDD_all_summer.loc[(
        (hhs_monthly_VCDD_all_summer["Months"]>=5) & 
        (hhs_monthly_VCDD_all_summer["Months"]<=9)),"Summer"] = "Yes"
    
    hhs_monthly_VCDD_all_summer.loc[~(
        (hhs_monthly_VCDD_all_summer["Months"]>=5) & 
        (hhs_monthly_VCDD_all_summer["Months"]<=9)),"Summer"] = "No"
    
    hhs_monthly_VCDD_all_summer.rename(columns = {"Labels":"Year_labels"}, inplace = True)
    
    hhs_monthly_VCDD_all_summer= hhs_monthly_VCDD_all_summer[
        hhs_monthly_VCDD_all_summer.Summer=="Yes"].groupby(
        ["Year_labels","models","acct"])["VCDD_RCP45", "VCDD_RCP85"].sum()
    
    #Old summer is referring to the short-run effects- only assumes temperature will change
    old_summer_energy = summertime_monthly_future.rename(
    columns = {"model":"models","Year":"Year_labels"}).set_index(
    ["acct","models","Year_labels"])
    
    #Creates a new df that has the slope and the monthly VCDD
    new_betas_summer_cooling = hhs_CDD_slopes.join(
        hhs_monthly_VCDD_all_summer,  on= ["Year_labels","models","acct"], 
        how= 'outer')
    
    new_betas_summer_cooling = new_betas_summer_cooling.join(
        old_summer_energy[['cool_kwh_RCP4.5','cool_kwh_RCP8.5','kwh_RCP4.5',
        'kwh_RCP8.5']], rsuffix = "_old")
    
    #Calculate the new cooling load CVDD(B_2_new)
    new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"] = new_betas_summer_cooling["RCP_85_new_betas"].multiply(
        new_betas_summer_cooling["VCDD_RCP85"])            
    
    #Calculate the new cooling load CVDD(B_2_new)            
    new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"] = new_betas_summer_cooling["RCP_45_new_betas"].multiply(
        new_betas_summer_cooling["VCDD_RCP45"])            
    
    #Calculates the new total electricity consumption---------------------- 
    # E_old - CVDD(B_2_old - B_2_new)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"]= new_betas_summer_cooling.loc[:,"kwh_RCP4.5"]-\
      new_betas_summer_cooling.loc[:,"VCDD_RCP45"]*(
    new_betas_summer_cooling.loc[:,"Original_Betas"].subtract(
        new_betas_summer_cooling.loc[:,"RCP_45_new_betas"]))
            
    new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"]= new_betas_summer_cooling.loc[:,"kwh_RCP8.5"]-\
      new_betas_summer_cooling.loc[:,"VCDD_RCP85"]*(
    new_betas_summer_cooling.loc[:,"Original_Betas"].subtract(
        new_betas_summer_cooling.loc[:,"RCP_85_new_betas"]))
    
    #--------Calculates the new total electricity cost----------------------     
    new_betas_summer_cooling.loc[:,"kwh_RCP8.5_cost_mean"]= new_betas_summer_cooling.loc[:,"kwh_RCP8.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"kwh_RCP8.5_cost_min"]= new_betas_summer_cooling.loc[:,"kwh_RCP8.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"kwh_RCP8.5_cost_max"]= new_betas_summer_cooling.loc[:,"kwh_RCP8.5"].multiply(annual_max_price)
    
    new_betas_summer_cooling.loc[:,"kwh_RCP4.5_cost_mean"]= new_betas_summer_cooling.loc[:,"kwh_RCP4.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"kwh_RCP4.5_cost_min"]= new_betas_summer_cooling.loc[:,"kwh_RCP4.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"kwh_RCP4.5_cost_max"]= new_betas_summer_cooling.loc[:,"kwh_RCP4.5"].multiply(annual_max_price)
    
    new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5_cost_mean"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5_cost_mean"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5_cost_min"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5_cost_min"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5_cost_max"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"].multiply(annual_max_price)
    new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5_cost_max"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"].multiply(annual_max_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5_cost_mean"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5_cost_min"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5_cost_max"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5"].multiply(annual_max_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5_cost_mean"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5"].multiply(annual_avg_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5_cost_min"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5"].multiply(annual_min_price)
    new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5_cost_max"]= new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5"].multiply(annual_max_price)
    #--------Calculates the new total electricity cost----------------------     
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP4.5_cost_mean']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"].multiply(summer_avg_price)
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP8.5_cost_mean']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"].multiply(summer_avg_price)
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP4.5_cost_min']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"].multiply(summer_min_price)
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP8.5_cost_min']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"].multiply(summer_min_price)
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP4.5_cost_max']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"].multiply(summer_max_price)
    new_betas_summer_cooling.loc[:,'new_cool_kwh_RCP8.5_cost_max']= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"].multiply(summer_max_price)
    
    #---------Calculates the percentage change between the new consumption and old (Total)       
    new_betas_summer_cooling.loc[:,"RCP4.5_SEER_per_change"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"].divide(
                new_betas_summer_cooling.loc[:,"kwh_RCP4.5"]).subtract(1).multiply(100)
    new_betas_summer_cooling.loc[:,"RCP8.5_SEER_per_change"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"].divide(
                new_betas_summer_cooling.loc[:,"kwh_RCP8.5"]).subtract(1).multiply(100) 
    
    #---------Calculates the percentage change between the new consumption and old (Cooling)       
    new_betas_summer_cooling.loc[:,"cool_RCP4.5_SEER_per_change"]= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"].divide(
                new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5"]).subtract(1).multiply(100)
    new_betas_summer_cooling.loc[:,"cool_RCP8.5_SEER_per_change"]= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"].divide(
                new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5"]).subtract(1).multiply(100) 
    
    #---------Calculates the relative changes total summertime consumption
    new_betas_summer_cooling.loc[:,"RCP4.5_SEER_kwh_change"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP4.5"].subtract(
                new_betas_summer_cooling.loc[:,"kwh_RCP4.5"])
                   
    new_betas_summer_cooling.loc[:,"RCP8.5_SEER_kwh_change"]= new_betas_summer_cooling.loc[:,"new_kwh_RCP8.5"].subtract(
                new_betas_summer_cooling.loc[:,"kwh_RCP8.5"])
            
    #---------Calculates the relative changes cooling summertime consumption
    new_betas_summer_cooling.loc[:,"cool_RCP4.5_SEER_kwh_change"]= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP4.5"].subtract(
                new_betas_summer_cooling.loc[:,"cool_kwh_RCP4.5"])
                   
    new_betas_summer_cooling.loc[:,"cool_RCP8.5_SEER_kwh_change"]= new_betas_summer_cooling.loc[:,"new_cool_kwh_RCP8.5"].subtract(
                new_betas_summer_cooling.loc[:,"cool_kwh_RCP8.5"])
    
    new_betas_summer_cooling = new_betas_summer_cooling.join(summertime_monthly_past.loc[:,[
          'cool_kwh_RCP8.5', 'cool_kwh_RCP4.5','kwh_RCP8.5', 'kwh_RCP4.5']],on=["acct","models",],  rsuffix= "_hist")
 
    new_betas_df_energy = add_demo_data_ele_sim(new_betas_summer_cooling.reset_index(), 
                                               past_data_annual.reset_index() )
    new_betas_df_energy.to_csv(path+"/Data/New_Beta_Estimates/Energy_estimates.csv")
    
    return()
new_beta_estimates()

#Do not load unless it has been uploded to Box (Double-Check upload)
new_betas_df_energy = pd.read_csv(path+"/Data/New_Beta_Estimates/Energy_estimates.csv",low_memory= True)
new_betas_df_energy.loc[:,"Year_s"] = new_betas_df_energy["Year_labels"].str.split("/").str[0]
new_betas_df_energy.loc[:,"Year_s"] = new_betas_df_energy.loc[:,"Year_s"].astype(int)

#Adjust identify the assumed and reported SEER ratings 
def adjust_seers(new_betas_df_energy):
    #Downloads table either assigns or uses survey SEER rating scores for the baseline model 
    comparison_table = pd.read_csv(path+"/Data/calibration_df.csv")
    comparison_table.rename(columns = {"BILACCT_K":"acct"}, inplace= True)
    comparison_table.set_index("acct", inplace=True)
    
    #Matches the actual SEER with the calibrated SEER 
    comparison_table.loc[~(comparison_table.VACSER1.isnull()),"Calibrated_SEER"] = \
        comparison_table.loc[~(comparison_table.VACSER1.isnull()),"VACSER1"]
        
    new_betas_df_energy= new_betas_df_energy.join(comparison_table[["Calibrated_SEER", 
                                                                    "SEER_labels"]],
                                    on="acct", 
                                    rsuffix= "_baseline")
    #Locates the SEER labels that are below the baseline SEER lable and set equal to zero
    new_betas_df_energy.loc[
        (new_betas_df_energy.SEER_labels <new_betas_df_energy.SEER_labels_baseline.round(0)),
        "Cal_list"]=0
    #Locate SEER labels that are identical to the baseline SEER labels and set them equal to the calibrated label
    new_betas_df_energy.loc[
        (new_betas_df_energy.SEER_labels ==new_betas_df_energy.SEER_labels_baseline.round(0)),
        "Cal_list"]=new_betas_df_energy.loc[
            (new_betas_df_energy.SEER_labels ==new_betas_df_energy.SEER_labels_baseline.round(0)),
            "Calibrated_SEER"].round(0)
    
    #Locate SEER labels that are less than the baseline SEER labels and subtract the baseline SEER labels from the LR
    new_betas_df_energy.loc[
        (new_betas_df_energy.SEER_labels >new_betas_df_energy.SEER_labels_baseline.round(0)),
        "Cal_list"]=new_betas_df_energy.loc[
            (new_betas_df_energy.SEER_labels >new_betas_df_energy.SEER_labels_baseline.round(0)),
            "Calibrated_SEER"].round(0).add(new_betas_df_energy.loc[
                (new_betas_df_energy.SEER_labels >new_betas_df_energy.SEER_labels_baseline.round(0)),
                "SEER_labels"].subtract(new_betas_df_energy.loc[
                    (new_betas_df_energy.SEER_labels >new_betas_df_energy.SEER_labels_baseline.round(0)),
                    "SEER_labels_baseline"]))
    return(new_betas_df_energy)

new_betas_df_energy = adjust_seers(new_betas_df_energy)
new_betas_df_energy.to_csv(path+"/Data/New_Beta_Estimates/Energy_estimates_cal.csv")
new_betas_df_energy = pd.read_csv(path+"/Data/New_Beta_Estimates/Energy_estimates_cal.csv")

#%% AC Efficiency Scenarios
#
def AC_efficiency_changes(SEER_yr1 = 15, SEER_yr2 = 18, SEER_yr3= 21,
                          save_on=True):
    #Create Year labels
    new_betas_df_energy.loc[:,"Year_s"] = new_betas_df_energy["Year_labels"].str.split("/").str[0]
    new_betas_df_energy.loc[:,"Year_s"] = new_betas_df_energy.loc[:,"Year_s"].astype(int)
    #Creates Decade labels as a category
    new_betas_df_energy.loc[:,"Decade"] = (
        np.floor(new_betas_df_energy.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype("category")
    #Changes the SEER labels into integers
    new_betas_df_energy.loc[:,"SEER_labels"] =new_betas_df_energy.loc[:,"SEER_labels"].astype(int)
    #Changes income labels into categories
    new_betas_df_energy["IG_num"] = new_betas_df_energy["IG_num"][~(
        new_betas_df_energy["IG_num"].isnull())].astype(int).astype("category")
    
    #Using a 20 year life-span of an AC unit
#------------------------------------------------------------------------------
    #Cooling 
#------------------------------------------------------------------------------  
    #Long run effects
    #Takes the average of the annual household model 
    reduction_over_the_years_cool = pd.concat([
        new_betas_df_energy[(
            (new_betas_df_energy["Year_s"]>=2020) & (new_betas_df_energy["Year_s"]<= 2039) &
            (new_betas_df_energy["Cal_list"].add(0)==SEER_yr1))].groupby(["acct","Year_s","models"]).median(
            ).reset_index(), 
        new_betas_df_energy[
            (new_betas_df_energy["Year_s"]>=2040) & (new_betas_df_energy["Year_s"]<= 2059) &
            (new_betas_df_energy["Cal_list"].add(3).round(0)==SEER_yr2)].groupby(["acct","Year_s","models"]).median(
            ).reset_index(),
        new_betas_df_energy[
            (new_betas_df_energy["Year_s"]>=2060) & (new_betas_df_energy["Year_s"]<= 2070) &
            (new_betas_df_energy["Cal_list"].add(6).round(0)==SEER_yr3)].groupby(["acct","Year_s","models"]).median(
            ).reset_index()],
                axis=0)[["acct","Year_s","models",
                         "new_cool_kwh_RCP8.5","new_cool_kwh_RCP4.5","cool_kwh_RCP8.5",
                        "cool_kwh_RCP4.5","cool_kwh_RCP8.5_hist","cool_kwh_RCP4.5_hist"]]
                
    reduction_over_the_years_cool.rename(columns= {"new_cool_kwh_RCP8.5":
                                                   "long_run_RCP8.5",
                                                   "new_cool_kwh_RCP4.5":
                                                   "long_run_RCP4.5",
                                                   "cool_kwh_RCP8.5":
                                                    "short_run_RCP8.5",
                                                   "cool_kwh_RCP4.5":
                                                    "short_run_RCP4.5",
                                                   "cool_kwh_RCP8.5_hist":
                                                      "historic_RCP8.5",
                                                   "cool_kwh_RCP4.5_hist":
                                                      "historic_RCP4.5"},
                                         inplace= True)
    
    #Long-Run Costs 
    reduction_over_the_years_cool.loc[:,"long_run_RCP8.5_cost_mean" ] = reduction_over_the_years_cool["long_run_RCP8.5"]*summer_avg_price
    reduction_over_the_years_cool.loc[:,"long_run_RCP8.5_cost_min"  ] = reduction_over_the_years_cool["long_run_RCP8.5"]*summer_min_price
    reduction_over_the_years_cool.loc[:,"long_run_RCP8.5_cost_max"  ] = reduction_over_the_years_cool["long_run_RCP8.5"]*summer_max_price
    
    reduction_over_the_years_cool.loc[:,"long_run_RCP4.5_cost_mean" ] = reduction_over_the_years_cool["long_run_RCP4.5"]*summer_avg_price
    reduction_over_the_years_cool.loc[:,"long_run_RCP4.5_cost_min"  ] = reduction_over_the_years_cool["long_run_RCP4.5"]*summer_min_price
    reduction_over_the_years_cool.loc[:,"long_run_RCP4.5_cost_max"  ] = reduction_over_the_years_cool["long_run_RCP4.5"]*summer_max_price

    reduction_over_the_years_cool.loc[:,"short_run_RCP8.5_cost_mean"] = reduction_over_the_years_cool["short_run_RCP8.5"]*summer_avg_price
    reduction_over_the_years_cool.loc[:,"short_run_RCP8.5_cost_min" ] = reduction_over_the_years_cool["short_run_RCP8.5"]*summer_min_price
    reduction_over_the_years_cool.loc[:,"short_run_RCP8.5_cost_max" ] = reduction_over_the_years_cool["short_run_RCP8.5"]*summer_max_price
    
    reduction_over_the_years_cool.loc[:,"short_run_RCP4.5_cost_mean"] = reduction_over_the_years_cool["short_run_RCP4.5"]*summer_avg_price
    reduction_over_the_years_cool.loc[:,"short_run_RCP4.5_cost_min" ] = reduction_over_the_years_cool["short_run_RCP4.5"]*summer_min_price
    reduction_over_the_years_cool.loc[:,"short_run_RCP4.5_cost_max" ] = reduction_over_the_years_cool["short_run_RCP4.5"]*summer_max_price
    
    #----------------------------------------------------------------------------
    #Using a 20 year lifespan for the (total consumption)
    #----------------------------------------------------------------------------
    #Long run effects
    reduction_over_the_years = pd.concat([
        new_betas_df_energy[(
            (new_betas_df_energy["Year_s"]>=2020) & (new_betas_df_energy["Year_s"]<= 2040) &
            (new_betas_df_energy["Cal_list"].add(0)==SEER_yr1))].groupby(["acct","Year_s","models"]).median(
            ).reset_index(), 
        new_betas_df_energy[
            (new_betas_df_energy["Year_s"]>=2041) & (new_betas_df_energy["Year_s"]<= 2061) &
            (new_betas_df_energy["Cal_list"].add(3)==SEER_yr2)].groupby(["acct","Year_s","models"]).median(
                ).reset_index(),
        new_betas_df_energy[
            (new_betas_df_energy["Year_s"]>=2062) & (new_betas_df_energy["Year_s"]<= 2070) &
            (new_betas_df_energy["Cal_list"].add(6)==SEER_yr3)].groupby(["acct","Year_s","models"]).median(
                ).reset_index()],
                axis=0)[["acct","Year_s","models",
                         "new_kwh_RCP8.5","new_kwh_RCP4.5","kwh_RCP8.5",
                        "kwh_RCP4.5","kwh_RCP8.5_hist","kwh_RCP4.5_hist"]]
                         
    reduction_over_the_years.rename(columns = 
        {"new_kwh_RCP8.5":"long_run_RCP8.5", "new_kwh_RCP4.5":"long_run_RCP4.5",
         "kwh_RCP8.5":"short_run_RCP8.5", "kwh_RCP4.5":"short_run_RCP4.5",
         "kwh_RCP8.5_hist":"historic_RCP8.5", "kwh_RCP4.5_hist":"historic_RCP4.5"},
        inplace= True)
    
    #Long-Run Costs 
    reduction_over_the_years.loc[:,"long_run_RCP_8.5_cost_mean"] = reduction_over_the_years["long_run_RCP8.5"]*summer_avg_price
    reduction_over_the_years.loc[:,"long_run_RCP_8.5_cost_min" ] = reduction_over_the_years["long_run_RCP8.5"]*summer_min_price
    reduction_over_the_years.loc[:,"long_run_RCP_8.5_cost_max" ] = reduction_over_the_years["long_run_RCP8.5"]*summer_max_price
    reduction_over_the_years.loc[:,"long_run_RCP_4.5_cost_mean"] = reduction_over_the_years["long_run_RCP4.5"]*summer_avg_price
    reduction_over_the_years.loc[:,"long_run_RCP_4.5_cost_min" ] = reduction_over_the_years["long_run_RCP4.5"]*summer_min_price
    reduction_over_the_years.loc[:,"long_run_RCP_4.5_cost_max" ] = reduction_over_the_years["long_run_RCP4.5"]*summer_max_price

    reduction_over_the_years.loc[:,"short_run_RCP_8.5_cost_mean"]  =  reduction_over_the_years["short_run_RCP8.5"]*summer_avg_price
    reduction_over_the_years.loc[:,"short_run_RCP_8.5_cost_min" ]  =  reduction_over_the_years["short_run_RCP8.5"]*summer_min_price
    reduction_over_the_years.loc[:,"short_run_RCP_8.5_cost_max" ]  =  reduction_over_the_years["short_run_RCP8.5"]*summer_max_price
    reduction_over_the_years.loc[:,"short_run_RCP_4.5_cost_mean"]  =  reduction_over_the_years["short_run_RCP4.5"]*summer_avg_price
    reduction_over_the_years.loc[:,"short_run_RCP_4.5_cost_min" ]  =  reduction_over_the_years["short_run_RCP4.5"]*summer_min_price
    reduction_over_the_years.loc[:,"short_run_RCP_4.5_cost_max" ]  =  reduction_over_the_years["short_run_RCP4.5"]*summer_max_price
      
    #Adds the socio-demographic data to the df
    reduction_over_the_years_cool_demo = add_demo_data_ele_sim(reduction_over_the_years_cool,
                                                          past_data_annual.reset_index())
    
    reduction_over_the_years_demo = add_demo_data_ele_sim(reduction_over_the_years,
                                                                past_data_annual.reset_index())
    
    if save_on == True:
        #Saves
        reduction_over_the_years_cool_demo.to_csv(path+\
        "/Data/New_Beta_Estimates/reduction_data_cool_"+str(SEER_yr1)+"_"+str(SEER_yr2) +"_"+ str(SEER_yr3)+".csv")
        reduction_over_the_years_demo.to_csv(path+\
        "/Data/New_Beta_Estimates/reduction_data_total_"+str(SEER_yr1)+"_"+str(SEER_yr2) +"_"+ str(SEER_yr3)+".csv")                           
    return(reduction_over_the_years_cool_demo,reduction_over_the_years_demo)

# Runs the analysis looking at certain efficiency adoption scenarios
#Uncertainty based on adoption scenarios
#Low-Efficieny adoption scenario (SEER 15/17/21)
[reduction_over_the_years_cool_15_18_21,
 reduction_over_the_years_total_15_18_21]= AC_efficiency_changes(
     SEER_yr1 = 15, SEER_yr2 = 18, SEER_yr3= 21, save_on=True)
     
#Moderate-Efficiency Scenario (SEER 19/22/25)
[reduction_over_the_years_cool_22_25_28,
 reduction_over_the_years_total_22_25_28]= AC_efficiency_changes(
     SEER_yr1 = 23, SEER_yr2 = 26, SEER_yr3= 29, save_on=True)      

#High-efficiency adoption scenario   
[reduction_over_the_years_cool_25_28_31,
 reduction_over_the_years_total_25_28_31]= AC_efficiency_changes(
     SEER_yr1 = 25, SEER_yr2 = 28, SEER_yr3= 31, save_on=False)
 
reduction_over_the_years_cool_15_18_21.loc[:,"Decade"] = (
     np.floor(reduction_over_the_years_cool_15_18_21.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype("category")
  
reduction_over_the_years_cool_22_25_28.loc[:,"Decade"] = (
     np.floor(reduction_over_the_years_cool_22_25_28.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype("category")

reduction_over_the_years_cool_25_28_31.loc[:,"Decade"] = (
     np.floor(reduction_over_the_years_cool_25_28_31.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype("category")

#Identifies the households that will see a reduction from the beginning to the end of the scenarios
def benefiting_households(scenario):
    """This function returns the account number of the homes that has a
    higher long-run consumption than their short-run consumption (1) and the
    demographics of the accounts that will not switch. Displays the 
    percentages from the group sample that will not switch"""
    
    switching_table_85 = scenario.groupby(
    ["acct","Year_s"]).sum()[["long_run_RCP8.5","short_run_RCP8.5"]]
    switching_table_85.loc[:,"switch?"] = 0
    
    switching_table_45 = scenario.groupby(
    ["acct","Year_s"]).sum()[["long_run_RCP4.5","short_run_RCP4.5"]]
    switching_table_45.loc[:,"switch?"] = 0
    
    #Assigns 1 if the short-run consumption is higher than the long-run consumption
    switching_table_85.loc[:,"switch?"] = switching_table_85.loc[:,
        "switch?"].mask(switching_table_85["short_run_RCP8.5"]>switching_table_85["long_run_RCP8.5"],1)
    
    switching_table_45.loc[:,"switch?"] = switching_table_45.loc[:,
        "switch?"].mask(switching_table_45["short_run_RCP4.5"]>switching_table_45["long_run_RCP4.5"],1)
    
    #identifies the households that 
    s_accounts_45 = switching_table_45.reset_index().groupby("acct").sum()["switch?"]
    
    households_that_will_benefit_from_scenario_45 = s_accounts_45.where(
        s_accounts_45==50).dropna().index.values
    
    s_accounts_85 = switching_table_85.reset_index().groupby("acct").sum()["switch?"]
    
    households_that_will_benefit_from_scenario_85 = s_accounts_85.where(
        s_accounts_85==50).dropna().index.values

    # survey_all["VACSER1"][survey_all.BILACCT_K.isin(accts_not_switching)].dropna().plot(kind="hist", bins=15)
    #Creates a table of the demographic data of the households that are not benefitting from the efficiency improvemnts 
   
    #Switched dataframe 
    
    return(households_that_will_benefit_from_scenario_45,
           households_that_will_benefit_from_scenario_85)

[accts_benefiting_45,
 accts_benefiting_85] = benefiting_households(
     scenario=reduction_over_the_years_cool_15_18_21)
     
pd.DataFrame(data=accts_benefiting_45).to_csv(path+"/Data/accts_benefiting_45.csv")            
pd.DataFrame(data=accts_benefiting_85).to_csv(path+"/Data/accts_benefiting_85.csv")            

     
[accts_benefiting_45_222528,
 accts_benefiting_85_222528] = benefiting_households(
     scenario=reduction_over_the_years_cool_22_25_28)
#%% Reductions from efficiency improvements
#What is the total reduction with the adoption scenario
#find the locations where the short-runs are lower than with efficient ACs

def long_run_effects_differences(data_input,
                                 SEER_yr1 = 15, SEER_yr2 = 18, SEER_yr3= 21,
                                 save_df= False):
    """Returns
    
    Percentage change and relative differences"""
    #Locates households where the short-run is smaller than the long-run
    
    #Calculates the percent change with new betas (Efficiency Percentage change)
    #for the entire period ( I am collapsing decadal and model effects)
    data_input = data_input[~(data_input["short_run_RCP8.5"].isnull())]
    #Long-run effects with AC efficiency changes
    reduction_per_chg_table =  pd.DataFrame(data_input.groupby(
    ["acct","Year_s","models"]).sum()["long_run_RCP8.5"].divide(
        data_input.groupby(["acct","Year_s","models"]).sum(
            )["historic_RCP8.5"]).subtract(1).multiply(100),
        columns= ["Efficiency_Changes_RCP8.5"])
    
    reduction_per_chg_table.loc[:,"Efficiency_Changes_RCP4.5"] =  data_input.groupby(
        ["acct","Year_s","models"]).sum()["long_run_RCP4.5"].divide(
            data_input.groupby(["acct","Year_s","models"]).sum(
                )["historic_RCP4.5"]).subtract(1).multiply(100)
    #short run effects 
    reduction_per_chg_table.loc[:,"short_run_RCP8.5"] =  data_input.groupby(
        ["acct","Year_s","models"]).sum()["short_run_RCP8.5"].divide(
            data_input.groupby(["acct","Year_s","models"]).sum(
                )["historic_RCP8.5"]).subtract(1).multiply(100) 
            
    reduction_per_chg_table.loc[:,"short_run_RCP4.5"] =  data_input.groupby(
        ["acct","Year_s","models"]).sum()["short_run_RCP4.5"].divide(
            data_input.groupby(["acct","Year_s","models"]).sum(
                )["historic_RCP4.5"]).subtract(1).multiply(100) 
    # Adds the demograhic data 
    

    #Consumption Changes
    #The difference between short-run scenario and long-run scenario
    reduction_consum_chg_table =  pd.DataFrame(data_input.groupby(
        ["acct","Year_s","models"]).sum()["short_run_RCP8.5"].subtract(data_input.groupby(["acct","Year_s","models"]).sum(
                )["long_run_RCP8.5"]), columns= ["Efficiency_Changes_RCP8.5"])
            
    reduction_consum_chg_table.loc[:,"Efficiency_Changes_RCP4.5"] =  pd.DataFrame(data_input.groupby(
        ["acct","Year_s","models"]).sum()["short_run_RCP4.5"].subtract(data_input.groupby(["acct","Year_s","models"]).sum(
                )["long_run_RCP4.5"]))
    
    #Adds the demographic data to the dataframe
    reduction_consum_chg_tab_demo = add_demo_data_ele_sim(reduction_consum_chg_table.reset_index(),
                                                       past_data_annual.reset_index())
    reduction_per_chg_tab_demo = add_demo_data_ele_sim(reduction_per_chg_table.reset_index(),
                                                       past_data_annual.reset_index())
    if save_df==True:
        reduction_consum_chg_tab_demo.to_csv(path + "/Results/short_and_long_run_consumption_change_"+str(SEER_yr1) +"_"+ str(SEER_yr2) +"_"+ str(SEER_yr3)+".csv")
        reduction_per_chg_tab_demo.to_csv(path + "/Results/short_and_long_run_per_change_"+str(SEER_yr1) +"_"+ str(SEER_yr2) +"_"+ str(SEER_yr3)+".csv")

    return(reduction_consum_chg_tab_demo,reduction_per_chg_tab_demo)

changes_per_change = pd.read_csv(path + "/Results/short_and_long_run_per_change_"+str(15) +"_"+ str(18) +"_"+ str(21)+".csv")

[changes_consump_change,
 changes_per_change] = long_run_effects_differences(reduction_over_the_years_cool_15_18_21,
                             SEER_yr1 = 15, SEER_yr2 = 18, SEER_yr3= 21, save_df=True) 
                                                    
[changes_consump_change_22_25_28,
 changes_per_change_22_25_28] = long_run_effects_differences(reduction_over_the_years_cool_22_25_28,
                             SEER_yr1 = 22, SEER_yr2 = 25, SEER_yr3= 28, save_df=False) 
                                                             
[changes_consump_change_25_28_31,
 changes_per_change_25_28_31] = long_run_effects_differences(reduction_over_the_years_cool_25_28_31,
                             SEER_yr1 = 20, SEER_yr2 = 23, SEER_yr3= 25, save_df=False) 
                                                             
                                                             
changes_per_change.loc[:,"Decade"] = (
    np.floor(changes_per_change.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype('category')
changes_per_change_22_25_28.loc[:,"Decade"] = (
    np.floor(changes_per_change_22_25_28.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype('category')
changes_per_change_25_28_31.loc[:,"Decade"] = (
    np.floor(changes_per_change_25_28_31.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype('category')

changes_per_change.loc[:,"IG_num"] = changes_per_change.loc[:,"IG_num"].astype('category')
changes_per_change_20_23_25.loc[:,"Decade"] = changes_per_change_20_23_25.loc[:,"Decade"].astype("category")
changes_per_change_25_28_31.loc[:,"Decade"] = (
    np.floor(changes_per_change_22_25_28.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype("category")
changes_per_change_22_25_28.loc[:,"IG_num"] = changes_per_change_22_25_28.loc[:,"IG_num"].astype('category')


changes_consump_change.loc[:,"Decade"] = (
    np.floor(changes_consump_change.loc[:,"Year_s"].astype(int)/10)*10).astype(int)
changes_consump_change.loc[:,"IG_num"] = changes_per_change.loc[:,"IG_num"].astype('category')
#%%


changes_per_change[changes_per_change.acct.isin(
accts_benefiting_85)].groupby(["acct","Year_s","models"]
).median()["Efficiency_Changes_RCP8.5"].median(level=2).quantile(q=[0.2,0.5,0.8])
                              
changes_consump_change[changes_per_change.acct.isin(
    accts_benefiting_85)].groupby(["acct","Year_s","models"]
    ).mean()["Efficiency_Changes_RCP8.5"].sum(level=[0,2]).median(level=1).quantile(q=[0.2,0.5,0.8])            

changes_consump_change[changes_per_change.acct.isin(
    accts_benefiting_85)].groupby(["acct","Year_s","models"]
    ).mean()["Efficiency_Changes_RCP8.5"].median(level=[0,2]).median(level=1).quantile(q=[0.2,0.5,0.8])            
              
accts_benefiting_45 = pd.read_csv(path +"/Data/accts_benefiting_45.csv")
accts_benefiting_85 = pd.read_csv(path +"/Data/accts_benefiting_85.csv")

def reduction_tables_benefitors(data_input = changes_per_change,acct_85 =accts_benefiting_85.iloc[:,1],
                                acct_45 =accts_benefiting_45.iloc[:,1],
                                summary_stat="50%"):

    #Efficiency Summary for RCP 8.5 (Long-Run)
    #Group by each demographic group
    per_who_benefit_from_reductions = pd.DataFrame(data= pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat]],
        axis=0))
    per_who_benefit_from_reductions.columns=["Efficiency_Changes_RCP8.5"]
    
    per_who_benefit_from_reductions.loc[:,"Count_85"] = pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age"]).nunique()["acct"]], axis=0)
        
    per_who_benefit_from_reductions.loc[:,"Short_run_RCP8.5"] = pd.concat([
    data_input[data_input.acct.isin(acct_85)].groupby(["IG_num",]).describe()["short_run_RCP8.5"][summary_stat],
    data_input[data_input.acct.isin(acct_85)].groupby(["Race",]).describe()["short_run_RCP8.5"][summary_stat],
    data_input[data_input.acct.isin(acct_85)].groupby(["Age",]).describe()["short_run_RCP8.5"][summary_stat]], axis=0)

    #Efficiency Difference for RCP 8.5 (Short-Run minus long-run) 
    per_who_benefit_from_reductions.loc[:,"Differences_RCP8.5"] = per_who_benefit_from_reductions.loc[:,"Short_run_RCP8.5"].subtract(
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP8.5"])
    
    #Efficiency Summary for RCP 4.5 (Long-Run)
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP4.5"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num",]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race",]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age",]).describe()["Efficiency_Changes_RCP4.5"][summary_stat]], axis=0)
    
    #Efficiency Summary for RCP 4.5 (Short-Run)
    per_who_benefit_from_reductions.loc[:,"Short_run_RCP4.5"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num",]).describe()["short_run_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race",]).describe()["short_run_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age",]).describe()["short_run_RCP4.5"][summary_stat]], axis=0)
    
    per_who_benefit_from_reductions.loc[:,"Count_45"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age"]).nunique()["acct"]], axis=0)
        
   #Efficiency Difference for RCP 4.5 (Short-Run minus long-run) 
    per_who_benefit_from_reductions.loc[:,"Differences_RCP4.5"] = per_who_benefit_from_reductions.loc[:,"Short_run_RCP4.5"].subtract(
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP4.5"])
    
    return(per_who_benefit_from_reductions)

def reduction_tables_benefitors_kwh(data_input,acct_85=accts_benefiting_85,
                                    acct_45=accts_benefiting_45,
                                    summary_stat="50%"):
    #Efficiency Summary for RCP 8.5 (Long-Run)
    #Group by each demographic group
    
    per_who_benefit_from_reductions = pd.DataFrame(data= pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat]]))
    per_who_benefit_from_reductions.columns=["Efficiency_Changes_RCP8.5"]
    
    per_who_benefit_from_reductions.loc[:,"Count_85"] = pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age"]).nunique()["acct"]], axis=0)
    #Efficiency Summary for RCP 4.5 (Long-Run)
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP4.5"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat]], axis=0)
    
    per_who_benefit_from_reductions.loc[:,"Count_45"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age"]).nunique()["acct"]], axis=0)
    
    return(per_who_benefit_from_reductions)

              
efficiency_reduction_table_per_base_case= reduction_tables_benefitors(
    data_input = changes_per_change,acct_85 =accts_benefiting_85.iloc[:,1],
                                    acct_45 =accts_benefiting_45.iloc[:,1],
                                    summary_stat="50%")

changes_consump_change[changes_per_change.acct.isin(
    accts_benefiting_85)].groupby(["acct","Year_s","models","IG_num"]
    ).mean(numeric_only=True)["Efficiency_Changes_RCP8.5"].sum(level=[0,2,3]).median(level=[0]).quantile(q=[0.2,0.5,0.8])            
              
efficiency_reduction_table_per_higher_case= reduction_tables_benefitors(
    changes_per_change_22_25_28,acct_85 =accts_benefiting_85_222528,
                                    acct_45 =accts_benefiting_45_222528,summary_stat="50%")


def SI_Table_7(data_input = changes_per_change,acct_85 =accts_benefiting_85.iloc[:,1],
                                acct_45 =accts_benefiting_45.iloc[:,1],
                                summary_stat="50%"):

    #Efficiency Summary for RCP 8.5 (Long-Run)
    #Group by each demographic group
    per_who_benefit_from_reductions = pd.DataFrame(data= pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num","Decade"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race","Decade"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age","Decade"]).describe()["Efficiency_Changes_RCP8.5"][summary_stat]],
        axis=0))
    per_who_benefit_from_reductions.columns=["Efficiency_Changes_RCP8.5"]
    
    per_who_benefit_from_reductions.loc[:,"Count_85"] = pd.concat([
        data_input[data_input.acct.isin(acct_85)].groupby(["IG_num","Decade"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Race","Decade"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_85)].groupby(["Age","Decade"]).nunique()["acct"]], axis=0)
        
    per_who_benefit_from_reductions.loc[:,"Short_run_RCP8.5"] = pd.concat([
    data_input[data_input.acct.isin(acct_85)].groupby(["IG_num","Decade"]).describe()["short_run_RCP8.5"][summary_stat],
    data_input[data_input.acct.isin(acct_85)].groupby(["Race","Decade"]).describe()["short_run_RCP8.5"][summary_stat],
    data_input[data_input.acct.isin(acct_85)].groupby(["Age","Decade"]).describe()["short_run_RCP8.5"][summary_stat]], axis=0)

    #Efficiency Difference for RCP 8.5 (Short-Run minus long-run) 
    per_who_benefit_from_reductions.loc[:,"Differences_RCP8.5"] = per_who_benefit_from_reductions.loc[:,"Short_run_RCP8.5"].subtract(
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP8.5"])
    
    #Efficiency Summary for RCP 4.5 (Long-Run)
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP4.5"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num","Decade"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race","Decade"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age","Decade"]).describe()["Efficiency_Changes_RCP4.5"][summary_stat]], axis=0)
    
    #Efficiency Summary for RCP 4.5 (Short-Run)
    per_who_benefit_from_reductions.loc[:,"Short_run_RCP4.5"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num","Decade"]).describe()["short_run_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race","Decade"]).describe()["short_run_RCP4.5"][summary_stat],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age","Decade"]).describe()["short_run_RCP4.5"][summary_stat]], axis=0)
    
    per_who_benefit_from_reductions.loc[:,"Count_45"] = pd.concat([
        data_input[data_input.acct.isin(acct_45)].groupby(["IG_num","Decade"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Race","Decade"]).nunique()["acct"],
        data_input[data_input.acct.isin(acct_45)].groupby(["Age","Decade"]).nunique()["acct"]], axis=0)
        
   #Efficiency Difference for RCP 4.5 (Short-Run minus long-run) 
    per_who_benefit_from_reductions.loc[:,"Differences_RCP4.5"] = per_who_benefit_from_reductions.loc[:,"Short_run_RCP4.5"].subtract(
    per_who_benefit_from_reductions.loc[:,"Efficiency_Changes_RCP4.5"])
    
    return(per_who_benefit_from_reductions)

SI_Table_7_df = SI_Table_7(data_input = changes_per_change,acct_85 =accts_benefiting_85.iloc[:,1],
                                acct_45 =accts_benefiting_45.iloc[:,1],
                                summary_stat="50%")
hh_data_per= pd.concat([
    data_input[data_input.acct.isin(accts_benefiting_45.iloc[:,1])].groupby(["Decade"]).nunique()["acct"],
    
    data_input[data_input.acct.isin(accts_benefiting_45.iloc[:,1])].groupby(["Decade"]).median()[["short_run_RCP4.5",
                                                                                                   "Efficiency_Changes_RCP4.5"]],
    data_input[data_input.acct.isin(accts_benefiting_85.iloc[:,1])].groupby(["Decade"]).nunique()["acct"],
    
    data_input[data_input.acct.isin(accts_benefiting_85.iloc[:,1])].groupby(["Decade"]).median()[["short_run_RCP8.5",
                                                                                                   "Efficiency_Changes_RCP8.5"]]],
    axis=1)  

SI_Table_7_final = SI_Table_7_df.join(hh_data_per, on="Decade",rsuffix="_all_hhs").round(1)
SI_Table_7_final.to_excel(path+"/Results/SI_Table_7.xlsx")
#Efficiency changes for consumption 

efficiency_reduction_table_kwh = reduction_tables_benefitors_kwh(
    changes_consump_change,accts_benefiting_85,accts_benefiting_45,summary_stat="50%")

efficiency_reduction_table_kwh_higher_case = reduction_tables_benefitors_kwh(
    changes_consump_change,accts_benefiting_85_222528,
    accts_benefiting_45_222528,summary_stat="50%")
[accts_benefiting_45_222528,
 accts_benefiting_85_222528,demo_of_benefactors] = benefiting_households(
     reduction_over_the_years_cool_22_25_28)

efficiecny_reduction_table_kwh_plot = efficiecny_reduction_table_kwh.reset_index().melt(
    id_vars= ["index"],value_vars=['Efficiency_Changes_RCP8.5',
                                            'Efficiency_Changes_RCP4.5'], 
    value_name = "Differences")

changes_per_change[changes_per_change.acct.isin(
accts_benefiting_85.index.values)].groupby(["IG_num","Year_s","models"]
).mean()[["Efficiency_Changes_RCP8.5"]].median(level=[0,2]).median(level=0).quantile(q=[0.2,0.5,0.8])

changes_per_change[changes_per_change.acct.isin(
accts_benefiting_85.index.values)].groupby(["Age","Year_s","models"]
).mean()[["Efficiency_Changes_RCP8.5"]].median(level=[0,2]).median(level=0).quantile(q=[0.2,0.5,0.8])

plot_testing = changes_per_change[changes_per_change.acct.isin(
accts_not_switching_85.index.values)].groupby(["IG_num","Year_s","models"]
).mean()[["Efficiency_Changes_RCP8.5","short_run_RCP4.5"]].reset_index().melt(
    id_vars= ["IG_num","Year_s","models"],
    value_vars=["Efficiency_Changes_RCP8.5","short_run_RCP4.5"],
    value_name = "kWh", 
    var_name= "Scenarios")
    
#Determines the averg  
accts_consumption = pd.DataFrame(changes_consump_change[changes_consump_change.acct.isin(
    accts_benefiting_85)].groupby(["acct","Year_s","models"]
    ).mean()["Efficiency_Changes_RCP8.5"].sum(level=["acct","models"])).reset_index().set_index("acct")          
accts_consumption.info()
accts_consumption = accts_consumption.join(changes_consump_change.set_index("acct").loc[:,["Race","Age","IG_num"]],
                                on ="acct")

accts_consumption.groupby(["IG_num","models"]).median().loc[1,:,:].quantile(q=[0.2,0.5,0.8])
accts_consumption.groupby(["Age","models"]).median().loc["65 yrs or older",:,:].quantile(q=[0.2,0.5,0.8])

accts_percent = pd.DataFrame(changes_per_change[changes_per_change.acct.isin(
    accts_benefiting_85)].groupby(["acct","Year_s","models"]
    ).median()["Efficiency_Changes_RCP8.5"].median(level=["acct","models"])
                                  ).reset_index().set_index("acct")          
accts_percent = accts_percent.join(changes_consump_change.set_index("acct").loc[:,["Race","Age","IG_num"]],
                                on ="acct")

accts_percent.groupby(["IG_num",]).median().loc[1,:,:].quantile(q=[0.2,0.5,0.8])
accts_percent.groupby(["Age"]).median().loc["65 yrs or older",:].quantile(q=[0.2,0.5,0.8])


changes_consump_change[changes_per_change.acct.isin(
    accts_that_benefit_85.index.values)].groupby(["Age","Year_s","models"]
    ).mean()["Efficiency_Changes_RCP8.5"].sum(level=[0,2]).loc["65 yrs or older",:,:].quantile(q=[0.2,0.5,0.8]) 

ax = sns.FacetGrid(data=plot_testing, col='IG_num',
                   col_wrap=2, height=4, aspect=1.2)
ax.map_dataframe(sns.boxplot, y="models",x="kWh" )
ax.set_titles("{col_name}")
ax.set_xlabels("")
ax.get_legend()
#plt.legend(loc="upper left")
plt.legend(bbox_to_anchor=(1.9,0.8))
#%% Figure 6: Perchange Changes realtive to short-run scenario
reduction_per_chg_tab_demo_19_22_25.loc[:,"Decade"] = (
    np.floor(reduction_per_chg_tab_demo_19_22_25.loc[:,"Year_s"].astype(int)/10)*10).astype(int)
reduction_per_chg_tab_demo_14_17_20.loc[:,"Decade"] = (
    np.floor(reduction_per_chg_tab_demo_14_17_20.loc[:,"Year_s"].astype(int)/10)*10).astype(int)

efficiency_reduction_table_per_19_22_25.loc[:,"Decade"] = efficiency_reduction_table_per_19_22_25.loc[:,"Decade"].astype("category")

plot_data_15_18_21 = changes_per_change.melt(id_vars=['IG_num','Race',"Year_s", "Decade",'Age',"acct","models"],
     value_vars=['Efficiency_Changes_RCP8.5', 'short_run_RCP8.5'],
     var_name = "Scenarios",
      value_name="Percentage_Change")
plot_data_15_18_21.to_excel(path+"/Data/New_Beta_Estimates/SER_15_18_21_plot_data.xlsx")

plot_data_22_25_28 = changes_per_change_22_25_28.melt(id_vars=['IG_num','Race', "Decade",'Age',"acct"],
     value_vars=['Efficiency_Changes_RCP8.5', 'short_run_RCP8.5'],
     var_name = "Scenarios",
      value_name="Percentage_Change")

def Figure_efficiency_point(plot_data, save_file=True,
                            all_hhs= False):
    """Returns 
    a pointplot of median percentage changes for demographic group.
    plot_data uses a column for percentage change, where the scenarios are labelled as short-run or long-run. """
    
    all_points = []
    if all_hhs==False:
        plot_data= plot_data[plot_data.acct.isin(accts_benefiting_85)]
    
    fig,axs= plt.subplots(3,1,dpi= 1000,sharex= True,figsize= (9,11),
                          constrained_layout= True)
    sns.pointplot( 
        ax=axs[2],
        data = plot_data[
       ((plot_data.Race == "White/Caucasian") |
        (plot_data.Race == "Asian") |
        (plot_data.Race == "Hispanic") | 
        (plot_data.Race == "Black or African American"))
        ]
        , y="Race", x ="Percentage_Change", palette= ["black", "green"],
        hue= "Scenarios",units="acct",
        seed=15,capsize = 0.25,errwidth=0.9, 
        markers=["s","^"], join=False,
        estimator= np.median,
        edgecolor= "black")
    
    for l in axs[2].lines:
        print(l.get_data())
    sns.pointplot(
        ax=axs[0],
        data = plot_data, units="acct",
        y="IG_num", x ="Percentage_Change", 
        palette=["black","green"],
        hue= "Scenarios",
        capsize = 0.25,seed=15,
        errwidth=0.9, 
        order =[1,2,3,4,5,6,7,8],
        errcolor= "black", estimator= np.median,
        markers=["s","^"], join=False)
    
    sns.pointplot(ax=axs[1], data = plot_data[
        (plot_data.Age !="")],
        errwidth=0.9, 
        y="Age",x ="Percentage_Change", palette= ["black", "green"],
        hue= "Scenarios",
        order = ['18-24 yrs old', '25-34 yrs old','35-44 yrs old', '45-54 yrs old',
                 '55-64 yrs old','65 yrs or older'],
        seed=15, markers=["s","^"], join=False,
        estimator= np.median,edgecolor= "black",  units="acct",capsize = 0.25)
    sns.despine(top= True, right= True)
    handles, labels = axs[0].get_legend_handles_labels()
    
    title_labels = list(string.ascii_lowercase[:3])
    axs[2].tick_params(axis='both', which='major', labelsize=15)
    for i in range(3):
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].set_title(
            ("(" + title_labels[i] + ")"), loc='left',
            fontname="Arial", fontsize=20, fontweight="bold")
        axs[i].get_legend().remove()
        labels = axs[i].get_yticklabels()
        
        #Saves the point estimates
        for path_graph in axs[i].collections:
            points = axs[i].collections[1].get_offsets()
            point_list=  pd.DataFrame(points)
        all_points.append(pd.DataFrame(point_list))    
        points_df =pd.concat(all_points,axis=0)
        if i == 0:
            axs[i].set_yticklabels(labels=income_group_numbers,
                                   fontsize=15)
        else:
            axs[i].set_yticklabels(labels=labels,
                                   fontsize=18 )
    fig.legend(loc= "lower center", handles= handles,
               labels = ["AC Efficiency + Temperature Changes", "Only Temperature Changes"],
               ncol= 4, fontsize= 16, 
               bbox_to_anchor =(0.6,-0.07),
               frameon= False)
    fig.supxlabel("Percentage Change Relative to Baseline (%)", fontsize= 18, 
                  fontweight= "bold")
    if save_file ==True:
        plt.savefig( path+'/Results/efficiency_improvement_pointplot.png', format='png',
                    dpi=1100,bbox_inches='tight')
    #     pd.concat(point_estimates_x,axis=0).to_csv(path+"/Results/point_estimates_graph_data_x.csv")
    #     pd.concat(point_estimates_y,axis=0).to_csv(path+"/Results/point_estimates_graph_data_y.csv")
    return(points_df)

Figure_efficiency_point(plot_data,save_file= True,
                        all_hhs= True)

Figure_efficiency_point(plot_data_14_17_20,save_file= False,
                        all_hhs= True)
Figure_efficiency_point(plot_data_15_18_21,save_file= True,
                        all_hhs= False)
Figure_efficiency_point(plot_data_22_25_28,save_file= False,
                        all_hhs= False)
#%%statistical testing: What are the p-values for the medians 
import scipy.stats as scripy
stat,p,med,tbl=scripy.median_test(plot_data_15_18_21[(
    plot_data_15_18_21["Scenarios"]=="Efficiency_Changes_RCP8.5") &
    (plot_data_15_18_21["IG_num"]==(1|2)) & 
    (plot_data_15_18_21.acct.isin(accts_benefiting_85))].groupby("acct").median()["Percentage_Change"].sort_values(),
    plot_data_15_18_21[(
    plot_data_15_18_21["Scenarios"]=="Efficiency_Changes_RCP8.5") &
    (plot_data_15_18_21["IG_num"]==7|8) &
    (plot_data_15_18_21.acct.isin(accts_benefiting_85))].groupby("acct").median()["Percentage_Change"].sort_values())



stat,p,med,tbl=scripy.median_test(plot_data_22_25_28[(
    plot_data_22_25_28["Scenarios"]=="Efficiency_Changes_RCP8.5") &
    (plot_data_22_25_28["IG_num"]==1) & 
    (plot_data_22_25_28.acct.isin(accts_benefiting_85_222528))].groupby("acct").median()["Percentage_Change"],
    plot_data_22_25_28[(
        plot_data_22_25_28["Scenarios"]=="Efficiency_Changes_RCP8.5") &
        (plot_data_22_25_28["IG_num"]==8) &
        (plot_data_22_25_28.acct.isin(accts_benefiting_85_222528))].groupby("acct").median()["Percentage_Change"])

stat,p_efficient_race,med,tbl=scripy.median_test(plot_data_15_18_21[(
    plot_data_15_18_21["Scenarios"]=="Efficiency_Changes_RCP8.5") &
    (plot_data_15_18_21["Race"]=="Black or African American") & 
    (plot_data_15_18_21.acct.isin(accts_benefiting_85))].groupby("acct").median()["Percentage_Change"],
    plot_data_15_18_21[(
        plot_data_15_18_21["Scenarios"]=="Efficiency_Changes_RCP8.5") &
        (plot_data_15_18_21["Race"]=="White/Caucasian") &
        (plot_data_15_18_21.acct.isin(accts_benefiting_85))].groupby("acct").median()["Percentage_Change"])

stat,p,med,tbl=scripy.median_test(plot_data[(
    plot_data["Scenarios"]=="Efficiency_Changes") &
    (plot_data["Race"]=="Black or African American")].groupby("acct").median()["Percentage_Change"],plot_data[(
        plot_data["Scenarios"]=="short_run") &
        (plot_data["Race"]=="Black or African American")].groupby("acct").median()["Percentage_Change"])
sns.utils.ci(
    sns.algorithms.bootstrap(np.array(plot_data_15_18_21.loc[(
    (plot_data_15_18_21.IG_num == 2) &
    (plot_data_15_18_21.Scenarios =="Efficiency_Changes_RCP8.5")),
    :].groupby(["acct","models"]).median()[col_name].dropna().values)))

sns.utils.ci(sns.algorithms.bootstrap(np.arange(100)))
def median_CI_income_race_age_sim(
        data_input=plot_data_15_18_21,
        col_name = "Percentage_Change",
        filtered_accts=accts_benefiting_85):
    """This function finds the confidence interval for a series and outputs
    each income group's respective estimate and confidence interval
    
    Input: dataframe with values and the income group numbers"""
    # --------------------------------------------------------------------------
    # Income
    # --------------------------------------------------------------------------
    data_input = data_input[data_input.acct.isin(filtered_accts)]
    df_income = pd.DataFrame(index=np.arange(1,9).astype(int),
                             columns=["Estimate", "LB", "UB"])
    for i in range(1,9):
        data_est = data_input.loc[(
            (data_input.IG_num == i) &
            (data_input.Scenarios =="Efficiency_Changes_RCP8.5")),
            :].groupby(["acct","models"]).median()[col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95, 
                        vectorized=True, axis=0, 
                        method='basic',  #Why does basic work by not bias?? read up on this
                        random_state=rng)
        df_income.loc[i,"Estimate"] = data_est.median()
        df_income.loc[i,["LB","UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Race
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_race = data_input.loc[(
            (data_input.Race != "") &
            (data_input.Race != "Native Hawaiian or Other") &
            (data_input.Race != "Pacific Islander")),:]
    # Defines a new dataframe for race estimates
    df_race = pd.DataFrame(index=data_input_race.Race.unique(),
                           columns=["Estimate","LB","UB"])
    for i, race in enumerate(df_race.index):
        data_est = data_input_race.loc[((data_input_race.Race == race) & 
                                        (data_input.Scenarios =="Efficiency_Changes_RCP8.5")),
                                       :].groupby(["acct","models"]).median()[col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic',vectorized=True,axis=0,
                        random_state=rng)
        df_race.loc[race, "Estimate"] = data_est.median()
        df_race.loc[race, ["LB","UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Age
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_age = data_input[
        (data_input.Age != "")]
    df_age = pd.DataFrame(index=data_input_age.Age.unique(),
                          columns=["Estimate", "LB","UB"])
    for i, age in enumerate(df_age.index):
        data_est = data_input_age.loc[((data_input_age.Age == age)&
                                       (data_input.Scenarios =="Efficiency_Changes_RCP8.5")),
                                      :].groupby(["acct","models"]).median()[col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic', vectorized=True,axis=0,
                        random_state=rng)
        df_age.loc[age, "Estimate"] = data_est.median()
        df_age.loc[age, ["LB", "UB"]] = res.confidence_interval
    
    return(df_income, df_race, df_age)

medians_eff = pd.concat(median_CI_income_race_age_sim(data_input=plot_data_15_18_21,
                                                    

medians_eff = pd.concat(median_CI_income_race_age_sim(
    data_input=plot_data_15_18_21,
    col_name = "Percentage_Change",
    filtered_accts=accts_benefiting_85),
    axis=0)
medians_eff.to_csv(path+  "/Results/point_plot_estimates_efficiency_15_18_21.csv")
