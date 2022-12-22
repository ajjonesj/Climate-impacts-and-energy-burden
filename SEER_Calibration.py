# -*- coding: utf-8 -*-
"""
SEER Estimator 
@author: Andrew Jones
"""

import pandas as pd
import numpy as np
from functions_and_labels import labels,filtered_hh, add_demo_data_ele_sim,add_race_income,add_demo_data_ele_sim
import matplotlib.pyplot as plt
import seaborn as sns
import string
import datetime as dt
plt.rcdefaults() 

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

#Removes the households with no shares because it means they do not have a 2017-2018 model
X_variables= X_models[~(X_models.Share.isnull())]
X_variables = X_variables.loc[:,['Intercept', 'Share', 'VFANNUM', 'VACSER1', 'AC_units_3 or more',
       'AC_units_Two', 'Central-Separate AC', 'Central-Unknown', 'VHOUSEHOLD',
       'VRESAGE', 'Sqft_1,500 - 2,999', 'Sqft_3,000 or more',
       'Apartment/Condo/Townhouse', 'Mobile home']] 
new_betas_hhs_all_original=[]
for seer_new in range(1,41):# Loops through the SEER ratings
    #Replaces the SEER rating with new SEER ratings
    X_variables_edit = X_variables[~(X_variables.VACSER1>seer_new)]

    X_variables_edit.loc[:,"VACSER1"] = seer_new
    #Creates a new df for the new betas
    Betas_df_hh_original = pd.DataFrame(
        CDD_slope_model.predict(X_variables_edit.dropna()),
        columns = ["New_Betas"])
    Betas_df_hh_original.loc[:,"SEER_labels"] = seer_new

    #Inserts a new columnd for the SEER ratings
    new_betas_hhs_all_original.append(Betas_df_hh_original)
#Saves the list into a dataframe    
new_betas_hhs_all_original_df = pd.concat(
    new_betas_hhs_all_original,
    axis=0)

#Adds the actual betas to the dataframe
baseline_betas_combination = new_betas_hhs_all_original_df.join(CDD_coeff.loc[:,"17/18"], on = "BILACCT_K")

#Calculates the absolute difference between the estimated and actual betas
baseline_betas_combination.loc[:,"ABS_Differences"] = np.abs(baseline_betas_combination.loc[:,"New_Betas"].subtract(
    baseline_betas_combination.loc[:,"17/18"]))

#Calculates the percentage error between the new and actual betas
baseline_betas_combination.loc[:,"Per_error"] = np.abs(baseline_betas_combination.loc[:,"New_Betas"].subtract(
    baseline_betas_combination.loc[:,"17/18"]).divide(
        baseline_betas_combination.loc[:,"17/18"]).multiply(100))
#Identifies the row where each account has the lowest percentage error 
lowest_difference = baseline_betas_combination.groupby("BILACCT_K")["Per_error"].min()
test = baseline_betas_combination
test = baseline_betas_combination.reset_index().groupby(["BILACCT_K","SEER_labels"]).mean()
test.min(level="BILACCT_K")

#Adds the minimum percentage change value to the dataframe 
SEER_comparison = baseline_betas_combination.join(lowest_difference,
                         on="BILACCT_K",rsuffix="_mins")

#Determines if the row values (Percentage error) matches the account's minimum percentage error
SEER_comparison.loc[:,"Compare"]= SEER_comparison.loc[:,
                                                        "Per_error"]==SEER_comparison.loc[:,"Per_error_mins"]

#A
accts_best_SEER = SEER_comparison[SEER_comparison.Compare ==True].groupby("BILACCT_K").mean()
actual_SEER = survey_all[["VACSER1","BILACCT_K"]].dropna()
actual_SEER.set_index("BILACCT_K",inplace=True)

    
comparison_table = accts_best_SEER.join(actual_SEER,on="BILACCT_K")



def calibration(): 
    comparison_table.loc[:,"Calibrated_SEER"]= np.nan
    comparison_table.loc[:,"Calibrated_?"] = np.nan
    acct_2_keep = comparison_table[comparison_table.SEER_labels ==comparison_table.VACSER1].index.values
    #keep the SEER
    comparison_table.loc[acct_2_keep,"Calibrated_SEER"]= comparison_table.loc[acct_2_keep,"SEER_labels"]
    comparison_table.loc[acct_2_keep,"Calibrated_?"]= "Not Calibrated"


    #identify households that did not report their SEER
    acct_2_change = comparison_table[comparison_table.SEER_labels !=comparison_table.VACSER1].index.values
    #Uses the calibration equation to change the SEER for households that did not report 
    comparison_table.loc[acct_2_change,"Calibrated_SEER"]=np.abs(
        0.0014*comparison_table.loc[acct_2_change,"SEER_labels"]**3 -\
            0.1138*comparison_table.loc[acct_2_change,"SEER_labels"]**2 +\
                2.8766*comparison_table.loc[acct_2_change,"SEER_labels"] - 8.162)
    comparison_table.loc[acct_2_change,"Calibrated_?"]= "Calibrated"
    comparison_table.to_csv(path+"/Data/calibration_df.csv")
comparison_table.dropna(inplace=True)
import scipy.stats as stats
stats.ttest_rel(a=comparison_table.dropna().iloc[:,0],
                b=comparison_table.dropna().iloc[:,1], alternative = 'two-sided')


add_race_income(comparison_table)

comparison_table.dropna().groupby("Race").mean()
comparison_table.dropna().groupby("IG_num").mean()
comparison_table.dropna().groupby("Age").mean()
comparison_table.max()
comparison_table.groupby("Race").mean()
comparison_table.groupby("IG_num").mean()
comparison_table.groupby("Age").mean()

comparison_table.groupby("IG_num").max()
