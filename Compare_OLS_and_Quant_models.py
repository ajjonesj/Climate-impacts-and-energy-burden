# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
five_pt_coeft_1718 = pd.read_excel(path+'/Data/Baseline/2017_2018/2017_2018_coefficients.xlsx',
    index_col=(0))
five_pt_coeft_1718.drop(axis=0,labels = ["BILACCT_K"], inplace=True)
temp_avgs = exogen.loc["05-01-2017":"04-30-2018"]["temp_avg"]
exogen_test = exogen.loc["05-01-2017":"04-30-2018"]
panel_data_test =  panel_data.loc["05-01-2017":"04-30-2018"]
temp_avgs = temp_avgs.groupby(temp_avgs.index.date).mean()
exogen_test.loc[:,"VCDD"] = np.nan
exogen_test.loc[:,"VHDD"] = np.nan
temp_response_df = pd.DataFrame(index = temp_avgs.index,
                                columns = five_pt_coeft_1718.index)

temp_OLS_groups = pd.DataFrame(index = temp_avgs.index,
                                columns = five_pt_coeft_1718.index)
panel_data_test.columns = five_pt_coeft_1718.index
for hh_count,house in enumerate(five_pt_coeft_1718.index):  
    #Dataframe of the accounts in the income group   
    daily_HDD_house = (annual_heating_points.loc[house,"17/18"]-temp_avgs).clip(lower=0)
    daily_CDD_house  = temp_avgs.subtract(annual_cooling_points.loc[house,"17/18"]).clip(lower=0)
    temp_response_df.loc[:,house] =   five_pt_coeft_1718.loc[house,"intercept"] + \
        five_pt_coeft_1718.loc[house,"HDD"]*daily_HDD_house+\
        five_pt_coeft_1718.loc[house,"CDD"]*daily_CDD_house
    exogen_test.loc[:,"VCDD"][exogen_test.BILACCT_K ==house] = daily_CDD_house 
    exogen_test.loc[:,"VHDD"][exogen_test.BILACCT_K ==house] = daily_HDD_house
exogen_test.to_csv(path+ '/Data/Baseline/exogen_variables_with_degree_days.csv')
ig_data_OLS= []
ig_data_Quant= []
ig_data_OLS[0]["VCDD",0]
#Income_Regressions
VCDD = pd.read_excel(path+'/Data/Baseline/household_VBCDD.xlsx')
VHDD = pd.read_csv(path+'/Data/Baseline/Individual_Heating_Balance_points.csv')

for ig_number in range(1,9):
    ig_num_accts =survey_all["BILACCT_K"][survey_all.VINCOME_new ==ig_number]
    panel_ig = panel_data_test.loc[:,
                                   panel_data_test.columns.isin(ig_num_accts.values)]
    panel_ig = panel_ig.stack().reset_index()
    panel_ig.columns = ["date_s","BILACCT_K","daily_kwh"]
    exogen_test_ig = exogen_test.loc[
                                   exogen_test.BILACCT_K.isin(ig_num_accts.values),:]
    exogen_test_ig = exogen_test_ig.reset_index()
    panel_data_ig = panel_ig.merge(exogen_test_ig,on=["date_s","BILACCT_K"])
    panel_data_ig = panel_data_ig.dropna()
    panel_data_ig_ln = panel_data_ig.loc[
        np.log(panel_data_ig.loc[:,"daily_kwh"]).index,:]
    
    panel_data_ig_ln.loc[:,"daily_kwh"] = np.log(panel_data_ig_ln.loc[:,"daily_kwh"])
    Y_model = panel_data_ig.loc[:,"daily_kwh"]
    X_model = sm.add_constant(panel_data_ig.loc[:,['holiday', 'dow_Friday',
           'dow_Monday', 'dow_Saturday', 'dow_Sunday', 'dow_Thursday',
           'dow_Tuesday',  'month_1', 'month_2', 'month_3',
            'month_5', 'month_6', 'month_7', 'month_8','month_9',
           'month_10', 'month_11', 'month_12', 'VCDD','VHDD']])
    OLS_mod = sm.OLS(Y_model,X_model,missing="drop",).fit(cov_type= 'cluster',
                                cov_kwds = {'groups':panel_data_ig["BILACCT_K"]})
    OLS_mod.summary()
    
    temp_coeff_table = pd.DataFrame(OLS_mod.params[["const","VCDD","VHDD"]], 
                                    columns = ["Betas"])
    temp_coeff_table.loc[:,"p_values"] = OLS_mod.pvalues[["const","VCDD","VHDD"]]
    temp_coeff_table.loc[:,["Betas_LB", "Betas_UB"]] =  pd.DataFrame(OLS_mod.conf_int(alpha=0.05).loc[["const","VCDD","VHDD"]])
                                  
    ig_data_OLS.append(
    Quant_mod= sm.QuantReg(Y_model,X_model).fit(cov_type= 'cluster',
                                cov_kwds = {'groups':panel_data_ig["BILACCT_K"]})
    Quant_mod.summary()
    ig_data_Quant.append(pd.concat([Quant_mod.params, Quant_mod.conf_int(alpha=0.05),Quant_mod.pvalues],axis=1))

    
           panel_ig,sm.add_constant(exogen_test.drop(["temp_avg","BILACCT_K"],axis=1))).fit(cov_type="cluster", 
                     