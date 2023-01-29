# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 03:05:19 2022

@author: Andrew Jones
"""

import pandas as pd
import numpy as np
from functions_and_labels import labels, filtered_hh, add_demo_data_ele_sim
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 1000

path = 'Insert path location of the project folder to save files here'
future_monthly_data = pd.read_csv(
    path + r'/Data/Consumption_Simulations/Monthly_estimates.csv')
historical_data = pd.read_csv(path + '/Data/Consumption_Simulations/historical.csv',
                              date_parser=('date_s'), low_memory=False)
past_data_annual = historical_data.groupby(["acct", "model"]).sum()
survey_all = pd.read_stata(
    'Insert survey data file here')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7, 4, 5, 6, 2, 1, 8, 3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"] != ""].unique()):
    survey_all.loc[survey_all["VINCOME"] == ig_num,
                   ["VINCOME_new"]] = int(ig_unsorted[idxs])
exogen= pd.read_csv(path+'/Data/exogen_data_mod.csv', parse_dates=[0],
                       index_col=[0])
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

survey_all_IG_tabl = survey_all.copy(deep=True).set_index("BILACCT_K")
IG_table = pd.DataFrame(data=survey_all_IG_tabl.VINCOME.str.split("to|than|or",
                                                                  expand=True,
                                                                  ))
IG_table.rename(columns={0: "Lower_Bound",
                         1: "Upper_Bound"}, inplace=True)
IG_table["Upper_Bound"] = IG_table["Upper_Bound"].str.replace(
    "[$,]", "", regex=True)
IG_table["Lower_Bound"] = IG_table["Lower_Bound"].str.replace(
    "[$,]", "", regex=True)
# Upper and lower bounds based on the reported income ranges (https://data.census.gov/cedsci/map?q=median%20income&t=Income%20and%20Poverty&g=0500000US04013%241400000&tid=ACSST5Y2020.S1901&cid=S1901_C01_011E&layer=VT_2020_140_00_PY_D1&mode=thematic&loc=30.1386,-109.6793,z6.2038)
IG_table["Lower_Bound"] = IG_table["Lower_Bound"].str.replace(
    "Less", "10000")
IG_table["Upper_Bound"] = IG_table["Upper_Bound"].str.replace(
    "m", "200000")
IG_table.drop(axis=1, labels=2, inplace=True)
IG_table_annual = IG_table[((IG_table.iloc[:, 0] != '') &
                            (IG_table.iloc[:, 1] != ''))].astype(float)
IG_table_annual.loc[:, "Median"] = IG_table_annual.median(axis=1)
IG_table_monthly = IG_table_annual/12
IG_table_monthly.rename(columns={"Upper_Bound": "Upper_Bound_monthly",
                                 "Lower_Bound": "Lower_Bound_monthly",
                                 "Median": "Median_monthly"}, inplace=True)
reduction_over_the_years_total_EB = pd.read_csv(
    path+"/Data/New_Beta_Estimates/reduction_data_total.csv").drop(axis=1, labels="Unnamed: 0")
new_betas_df_energy = pd.read_csv(path+"/Data/New_Beta_Estimates/Energy_estimates_cal.csv")

new_betas_df_energy.drop(["Unnamed: 0.1"], axis=1, inplace= True)

def cleaning_betas_df_costs(data_input):
    """Returns
    
    Percentage change and relative differences"""
    #Locates households where the short-run is smaller than the long-run
    loc_to_change_RCP85_kwh = data_input.loc[:,"new_kwh_RCP8.5"]>\
                                  data_input.loc[:,"kwh_RCP8.5"]
                                  
    loc_to_change_RCP45_kwh = data_input.loc[:,"new_kwh_RCP4.5"]>\
                                   data_input.loc[:,"kwh_RCP4.5"]
    
    loc_to_change_RCP85_cool = data_input.loc[:,"new_cool_kwh_RCP8.5"]>\
                                  data_input.loc[:,"cool_kwh_RCP8.5"]
                                  
    loc_to_change_RCP45_cool = data_input.loc[:,"new_cool_kwh_RCP4.5"]>\
                                   data_input.loc[:,"cool_kwh_RCP4.5"]
                                                      
    data_input.loc[loc_to_change_RCP85_kwh,"new_kwh_RCP8.5"]=\
        data_input.loc[loc_to_change_RCP85_kwh,
                       "kwh_RCP8.5"]
    
    data_input.loc[loc_to_change_RCP45_kwh,"new_kwh_RCP4.5"]=\
        data_input.loc[loc_to_change_RCP45_kwh,
                       "kwh_RCP4.5"]
        
    data_input.loc[loc_to_change_RCP85_cool,"new_cool_kwh_RCP8.5"]=\
        data_input.loc[loc_to_change_RCP85_cool,
                       "kwh_RCP8.5"]
        
    data_input.loc[loc_to_change_RCP45_cool,"new_cool_kwh_RCP4.5"]=\
        data_input.loc[loc_to_change_RCP45_cool,
                       "kwh_RCP4.5"]
    
    return(data_input)

new_betas_df_energy_new = cleaning_betas_df_costs(data_input = new_betas_df_energy)

def calculate_elec_costs(new_betas_df_energy):
    new_betas_df_energy.loc[:,"kwh_RCP8.5_cost_mean"]= new_betas_df_energy.loc[:,"kwh_RCP8.5"].multiply(annual_avg_price)
    new_betas_df_energy.loc[:,"kwh_RCP8.5_cost_min"]= new_betas_df_energy.loc[:,"kwh_RCP8.5"].multiply(annual_min_price)
    new_betas_df_energy.loc[:,"kwh_RCP8.5_cost_max"]= new_betas_df_energy.loc[:,"kwh_RCP8.5"].multiply(annual_max_price)
    new_betas_df_energy.loc[:,"kwh_RCP4.5_cost_mean"]= new_betas_df_energy.loc[:,"kwh_RCP4.5"].multiply(annual_avg_price)
    new_betas_df_energy.loc[:,"kwh_RCP4.5_cost_min"]= new_betas_df_energy.loc[:,"kwh_RCP4.5"].multiply(annual_min_price)
    new_betas_df_energy.loc[:,"kwh_RCP4.5_cost_max"]= new_betas_df_energy.loc[:,"kwh_RCP4.5"].multiply(annual_max_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP8.5_cost_mean"]= new_betas_df_energy.loc[:,"cool_kwh_RCP8.5"].multiply(annual_avg_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP8.5_cost_min"]= new_betas_df_energy.loc[:,"cool_kwh_RCP8.5"].multiply(annual_min_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP8.5_cost_max"]= new_betas_df_energy.loc[:,"cool_kwh_RCP8.5"].multiply(annual_max_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP4.5_cost_mean"]= new_betas_df_energy.loc[:,"cool_kwh_RCP4.5"].multiply(annual_avg_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP4.5_cost_min"]= new_betas_df_energy.loc[:,"cool_kwh_RCP4.5"].multiply(annual_min_price)
    new_betas_df_energy.loc[:,"cool_kwh_RCP4.5_cost_max"]= new_betas_df_energy.loc[:,"cool_kwh_RCP4.5"].multiply(annual_max_price)
    return(new_betas_df_energy)

new_betas_df_energy = calculate_elec_costs(new_betas_df_energy)
# %% Energy Burden

# %% New Energy Burden

def calculate_energy_burden(new_betas_df_energy):
    new_betas_df_energy_EB = new_betas_df_energy.join(
            IG_table_monthly.multiply(5), on="acct")
    
    new_betas_df_energy_EB.loc[:,"Monthly_EB_LB_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP8.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:,"Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP8.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:,"Median_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP8.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    # Only for AC use
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_cool_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP8.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:, "Lower_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_cool_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP8.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_cool_RCP85_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP8.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
    
    #Short_run
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_RCP85_SR"] = \
        new_betas_df_energy_EB.loc[:, "kwh_RCP8.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_RCP85_SR"] = \
        new_betas_df_energy_EB.loc[:, "kwh_RCP8.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_RCP85_Sr"] = \
        new_betas_df_energy_EB.loc[:,"kwh_RCP8.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    # Only for AC use
    new_betas_df_energy_EB.loc[:,"Monthly_EB_UB_cool_RCP85_SR"] = \
        new_betas_df_energy_EB.loc[:,"cool_kwh_RCP8.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_cool_RCP85_SR"] = \
        new_betas_df_energy_EB.loc[:, "cool_kwh_RCP8.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_cool_RCP85_SR"] = \
        new_betas_df_energy_EB.loc[:, "cool_kwh_RCP8.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
    
    new_betas_df_energy_EB.loc[:,"Monthly_EB_LB_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP4.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:,"Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP4.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:,"Median_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:,"new_kwh_RCP4.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    # Only for AC use
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_cool_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP4.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:, "Lower_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_cool_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP4.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_cool_RCP45_LR"] = \
        new_betas_df_energy_EB.loc[:, "new_cool_kwh_RCP4.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
    
    #Short_run
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_RCP45_SR"] = \
        new_betas_df_energy_EB.loc[:, "kwh_RCP4.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_RCP45_SR"] = \
        new_betas_df_energy_EB.loc[:, "kwh_RCP4.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_UB_RCP45_Sr"] = \
        new_betas_df_energy_EB.loc[:,"kwh_RCP4.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    # Only for AC use
    new_betas_df_energy_EB.loc[:,"Monthly_EB_UB_cool_RCP45_SR"] = \
        new_betas_df_energy_EB.loc[:,"cool_kwh_RCP4.5_cost_min"] /\
        new_betas_df_energy_EB.loc[:,"Lower_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_LB_cool_RCP45_SR"] = \
        new_betas_df_energy_EB.loc[:, "cool_kwh_RCP4.5_cost_mean"] /\
        new_betas_df_energy_EB.loc[:, "Upper_Bound_monthly"]*100
    
    new_betas_df_energy_EB.loc[:, "Monthly_EB_Median_cool_RCP45_SR"] = \
        new_betas_df_energy_EB.loc[:, "cool_kwh_RCP4.5_cost_max"] /\
        new_betas_df_energy_EB.loc[:, "Median_monthly"]*100
        
    
        
    return(new_betas_df_energy_EB)

new_betas_df_energy_EB = calculate_energy_burden(new_betas_df_energy)
new_betas_df_energy_EB.to_csv(path+"/Results/energy_bruden_cal.csv")
new_betas_df_energy_EB = pd.read_csv(
    path + '/Results/energy_bruden_cal.csv')

new_betas_df_energy_EB["Year"] = new_betas_df_energy_EB["Year_labels"].str.split(
    "/").str[0]
new_betas_df_energy_EB.loc[:,
                           "Year"] = new_betas_df_energy_EB.loc[:, "Year"].astype(int)
new_betas_df_energy_EB.loc[:, "IG_num"] = new_betas_df_energy_EB.loc[:, "IG_num"].astype(
    "category")
new_betas_df_energy_EB.loc[:,"Decade"] = (
    np.floor(new_betas_df_energy_EB.loc[:,"Year"].astype(int)/10)*10).astype(int)

new_betas_df_energy_EB[(new_betas_df_energy_EB.acct==22626003) & 
                       (new_betas_df_energy_EB.Decade==2030)].groupby(["models","Year_s", "Cal_list"]).mean()[["Monthly_EB_Median_RCP85_SR"]]
# %% Finalized figure for energy burden
# -------------------------------------------------------------------------------
def Energy_Burden_fig_benefitors_only(save_fig= False, 
                      all_hhs=False,
                      input_data = new_betas_df_energy_EB):

        fig,axs = plt.subplots(2,1, dpi=600, figsize=(8,6.5),
                                constrained_layout=True, sharex=True)
        axs = axs.ravel()
        sns.lineplot(
            ax=axs[0],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==15))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==18))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==21))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            ls="-", color="black", label="No SEER Changes",
            x="Year", y="Monthly_EB_Median_RCP85_SR",markers= True,
            err_kws={"alpha": 0},
            estimator=np.median, ci=95)

        sns.lineplot(
            ax=axs[0],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==15))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==18))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==21))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            x="Year", y="Monthly_EB_Median_RCP85_LR",markers= True,
            err_kws={"alpha": 0},
            estimator=np.median, ci=95,
            label="AC Efficiency Upgrades (15,18,21)",
            color="black", ls=":")
        sns.lineplot(
            ax=axs[0],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==23))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==26))],
                    input_data[(
                    (input_data.IG_num==1) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==29))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            x="Year", y="Monthly_EB_Median_RCP85_LR",
             ls="-.",
            err_kws={"alpha": 0},
            estimator=np.median, ci=95,
            label="AC Efficiency Upgrades (23,26,29)",
            color="black")
                    
        axs[0].fill_between(input_data[
        (input_data.IG_num==1)].reset_index()["Year"].unique(),
        y1 =pd.concat([      
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2020) &
                 (input_data["Year"]<= 2040) & 
                 (input_data["Cal_list"]==5))],
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2041) &
                 (input_data["Year"]<= 2061) & 
                 (input_data["Cal_list"]==5))],
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2062) &
                 (input_data["Year"]<= 2070) & 
                 (input_data["Cal_list"]==5))]],
                axis=0).groupby(["acct",
                     "Year"]
                    ).median().reset_index()["Monthly_EB_Median_RCP85_LR"].median(),
        y2=pd.concat([      
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2020) &
                 (input_data["Year"]<= 2040) & 
                 (input_data["Cal_list"]==40))],
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2041) &
                 (input_data["Year"]<= 2061) & 
                 (input_data["Cal_list"]==40))],
                input_data[(
                (input_data.IG_num==1) & 
                (input_data["Year"]>=2062) &
                 (input_data["Year"]<= 2070) & 
                 (input_data["Cal_list"]==40))]],
                axis=0).groupby(["acct",
                     "Year"]
                    ).median().reset_index()["Monthly_EB_Median_RCP85_LR"].median(),
                    color = "gray", alpha = 0.1, linewidth=1.5)
            
        sns.lineplot(
            ax=axs[1],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==15))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==18))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==21))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            ls="-", color="black", label="No SEER Changes",
            x="Year", y="Monthly_EB_Median_RCP85_SR",markers= True,
            err_kws={"alpha": 0},
            estimator=np.median, ci=95)
        
        sns.lineplot(
            ax=axs[1],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==15))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==18))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==21))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            x="Year", y="Monthly_EB_Median_RCP85_LR",markers= True,
            err_kws={"alpha": 0},
            estimator=np.median, ci=95,
            label="AC Efficiency Upgrades (15,18,21)",
            color="black", ls=":")
        sns.lineplot(
            ax=axs[1],
            data=pd.concat([      
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2020) &
                     (input_data["Year"]<= 2040) & 
                     (input_data["Cal_list"]==23))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2041) &
                     (input_data["Year"]<= 2061) & 
                     (input_data["Cal_list"]==26))],
                    input_data[(
                    (input_data.IG_num==3) & 
                    (input_data["Year"]>=2062) &
                     (input_data["Year"]<= 2070) & 
                     (input_data["Cal_list"]==29))]],
                    axis=0).groupby([
                        "acct", "Year"]
                        ).median().reset_index(),
            x="Year", y="Monthly_EB_Median_RCP85_LR",
             ls="-.",
            err_kws={"alpha": 0},
            estimator=np.median, ci=95,
            label="AC Efficiency Upgrades (23,26,29)",
            color="black")
                    
        axs[1].fill_between(input_data[
        (input_data.IG_num==3)].reset_index()["Year"].unique(),
        y1 =pd.concat([      
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2020) &
                 (input_data["Year"]<= 2040) & 
                 (input_data["Cal_list"]==5))],
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2041) &
                 (input_data["Year"]<= 2061) & 
                 (input_data["Cal_list"]==5))],
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2062) &
                 (input_data["Year"]<= 2070) & 
                 (input_data["Cal_list"]==5))]],
                axis=0).groupby(["acct",
                     "Year"]
                    ).median().reset_index()["Monthly_EB_Median_RCP85_LR"].median(),
        y2=pd.concat([      
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2020) &
                 (input_data["Year"]<= 2040) & 
                 (input_data["Cal_list"]==40))],
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2041) &
                 (input_data["Year"]<= 2061) & 
                 (input_data["Cal_list"]==40))],
                input_data[(
                (input_data.IG_num==3) & 
                (input_data["Year"]>=2062) &
                 (input_data["Year"]<= 2070) & 
                 (input_data["Cal_list"]==40))]],
                axis=0).groupby(["acct","Year"]).median().reset_index()["Monthly_EB_Median_RCP85_LR"].median(),
                        color = "gray", alpha = 0.1, linewidth=1.5, 
                        label = "SEER 5 to SEER 40")
            
            
        for i in range(2):
            axs[i].axhline(10, color="red", ls="--", linewidth=1,
                           label="Severe Energy Burden (>10%)", alpha=0.4)
            axs[i].axhline(6, color="red", ls="-", linewidth=1,
                           label="High Energy Burden (>6%)", alpha=0.4)
            axs[i].set_ylabel("%")
            axs[i].set_xlabel("Year")
    
            axs[i].set_ylim(0, 20)
            sns.despine(ax=axs[i], top=True, right=True)
            axs[i].get_legend().remove()
        
        axs[0].set_title("Income Group 1", fontweight="bold",
                         fontsize=12)
        axs[1].set_title("Income Group 3", fontweight="bold",
                         fontsize=12)
      
        
        handles, labels2 = axs[1].get_legend_handles_labels()
        fig.legend(loc="lower center",
                   ncol=2, fontsize=10, handles=handles,
                   bbox_to_anchor=(0.5, -0.1),
                   frameon=False)
        fig.supylabel("Summertime Energy Burden ",
                      fontsize=15, fontweight="bold")
        if save_fig==True:
            plt.savefig(path+"/Results/SEER_and_EB_V2_benefitors", dpi=1200,
                    bbox_inches='tight')


Energy_Burden_fig_benefitors_only(save_fig= True, all_hhs=False,
                                 input_data = new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)])
EB_SR=pd.concat([      
        new_betas_df_energy_EB[(
        
        (new_betas_df_energy_EB["Year"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==15))],
        new_betas_df_energy_EB[(
         
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==18))],
        new_betas_df_energy_EB[(
     
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==21))]],
        axis=0)[["acct","Monthly_EB_Median_RCP85_SR","Monthly_EB_LB_RCP85_SR",
                 "Monthly_EB_UB_RCP85_Sr","Monthly_EB_Median_RCP85_LR",
                 "Monthly_EB_LB_RCP85_LR","Year_s",
                 "Monthly_EB_UB_RCP85_LR","Age","IG_num","Race","models"]]



EB_151821=pd.concat([      
        new_betas_df_energy_EB[        new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)][(
        
        (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]>=2020) &
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]<= 2040) & 
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Cal_list"]==15))],
        new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)][(
         
        (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]>=2041) &
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]<= 2061) & 
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Cal_list"]==18))],
        new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)][(
     
        (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]>=2062) &
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Year"]<= 2070) & 
         (new_betas_df_energy_EB[
                                     new_betas_df_energy_EB.acct.isin(
                                         accts_benefiting_85)]["Cal_list"]==21))]],
        axis=0)[["acct","Monthly_EB_Median_RCP85_SR","Monthly_EB_LB_RCP85_SR",
                 "Monthly_EB_UB_RCP85_Sr","Monthly_EB_Median_RCP85_LR",
                 "Monthly_EB_LB_RCP85_LR","Year_s",
                 "Monthly_EB_UB_RCP85_LR","Age","IG_num","Race","models"]]



EB_151821.groupby(["acct","IG_num","models"]).mean().median(level=["IG_num"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]]
EB_151821.groupby(["acct","IG_num","models"]).mean().median(level=["IG_num","models"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]].groupby("IG_num").quantile(q=[0.2,0.5,0.8])
EB_151821[EB_151821["Monthly_EB_LB_RCP85_LR"]>=6].groupby(["IG_num","Year_s","models"]).nunique()["acct"].sum(level=["Year_s","IG_num","models"]).mean(level=["IG_num","models"]).reset_index().groupby("IG_num").quantile(q=0.5)-\
EB_151821[EB_151821["Monthly_EB_LB_RCP85_SR"]>=6].groupby(["IG_num","Year_s","models"]).nunique()["acct"].sum(level=["Year_s","IG_num","models"]).mean(level=["IG_num","models"]).reset_index().groupby("IG_num").quantile(q=0.5)

EB_151821[EB_151821["Monthly_EB_Median_RCP85_LR"]>=6].groupby(["Race","Year_s"]).nunique()["acct"].mean(level=["Race"])/\
EB_151821.groupby(["Race"]).nunique()["acct"]

(EB_SR[EB_SR["Monthly_EB_Median_RCP85_SR"]>=6].groupby(["Race","Year_s"]).nunique()["acct"].mean(level=["Race"])/\
EB_SR.groupby(["Race"]).nunique()["acct"])*100

EB_SR[EB_SR["Monthly_EB_Median_RCP85_SR"]>=6].groupby(["Race","Year_s"]).nunique()["acct"].sum(level=["Race"])/\
EB_SR.groupby(["Race"]).nunique()["acct"]

EB_SR[EB_SR["Monthly_EB_LB_RCP85_SR"]>=6].groupby(["Race","Year_s"]).nunique()["acct"].sum(level=["Race"])/\
EB_SR.groupby(["Race"]).nunique()["acct"]

EB_151821[EB_151821["Monthly_EB_UB_RCP85_Sr"]>=6].groupby(["IG_num","Year_s","models"]).nunique()["acct"].sum(level=["Year_s","IG_num"]).mean(level=["IG_num"])

EB_151821.groupby(["acct","IG_num","models"]).mean().median(level=["IG_num"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]]
EB_151821.groupby(["acct","Race","models"]).mean().quantile(level=["Race"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]]
EB_151821.groupby(["acct","Age","models"]).mean().median(level=["Age"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]]

sns.boxplot(data = EB_151821.groupby(["acct","IG_num","models","Year_s"]).mean().median(level=["IG_num","acct","Year_s"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]].reset_index(),
            x= "IG_num", y= "Monthly_EB_Median_RCP85_LR")
sns.violinplot(data = EB_151821.groupby(["acct","IG_num","models","Year_s"]).mean().median(level=["IG_num","acct","Year_s"])[["Monthly_EB_Median_RCP85_LR",
                                                                               "Monthly_EB_Median_RCP85_SR"]].reset_index(),
            x= "IG_num", y= "Monthly_EB_Median_RCP85_SR")

def energy_burden_SR_LR_table(new_betas_df_energy_EB, 
                              apt_yer_1 =15, 
                              apt_yer_2 =18,apt_yer_3 = 21, 
                              energy_burden_LR_scen ="Monthly_EB_Median_RCP85_LR",
                              energy_burden_SR_scen ="Monthly_EB_Median_RCP85_SR", 
                              burden_threshold= 6,all_hhs=False):
    
    table = pd.concat([
        #Long-run 
        pd.concat([
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["IG_num",]).nunique()["acct"],
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["Age",]).nunique()["acct"],
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_LR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["Race",]).nunique()["acct"]],axis=0),
        
        pd.concat([
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["IG_num",]).nunique()["acct"],
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["Age",]).nunique()["acct"],
        pd.concat([      
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
            (new_betas_df_energy_EB["Decade"]>=2020) &
         (new_betas_df_energy_EB["Year"]<= 2040) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2041) &
         (new_betas_df_energy_EB["Year"]<= 2061) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
        new_betas_df_energy_EB[((new_betas_df_energy_EB[energy_burden_SR_scen]>burden_threshold) & 
        (new_betas_df_energy_EB["Year"]>=2062) &
         (new_betas_df_energy_EB["Year"]<= 2070) & 
         (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
        axis=0).groupby(["Race",]).nunique()["acct"]],axis=0)
    ],axis=1)
    
    table.columnns  = ["Long_Run", "Short_Run"]
    
    EB_accts = pd.concat([
     pd.concat([new_betas_df_energy_EB[(
        (new_betas_df_energy_EB["Decade"]>=2020) &
     (new_betas_df_energy_EB["Year"]<= 2040) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
    new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2041) &
     (new_betas_df_energy_EB["Year"]<= 2061) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
    new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2062) &
     (new_betas_df_energy_EB["Year"]<= 2070) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
    axis=0).groupby(["acct",]).median()[energy_burden_LR_scen],
    pd.concat([new_betas_df_energy_EB[(
        (new_betas_df_energy_EB["Decade"]>=2020) &
     (new_betas_df_energy_EB["Year"]<= 2040) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_1))],
    new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2041) &
     (new_betas_df_energy_EB["Year"]<= 2061) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_2))],
    new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2062) &
     (new_betas_df_energy_EB["Year"]<= 2070) & 
     (new_betas_df_energy_EB["Cal_list"]==apt_yer_3))]],
    axis=0).groupby(["acct",]).median()[energy_burden_SR_scen]],axis=1)
    
    EB_accts.columnns  = ["Long_Run", "Short_Run"]
    return(table)

compar_tab = energy_burden_SR_LR_table(new_betas_df_energy_EB, 
                                       apt_yer_1 =24, 
                                       apt_yer_2=27,
                                       apt_yer_3 = 30, 
                              energy_burden_LR_scen ="Monthly_EB_Median_RCP85_LR",
                              energy_burden_SR_scen ="Monthly_EB_Median_RCP85_SR",
                              burden_threshold= 6)
compar_tab.columns = ["Long_Run", "Short_Run"]
compar_tab.loc[:,"Difference"]= compar_tab.loc[:,"Short_Run"].subtract(compar_tab.loc[:,"Long_Run"])

sns.barplot(data = compar_tab.reset_index(),
            y="index", x="Difference", color="gray")
sns.despine(top= True, right=True)
plt.axvline(0, linewidth=2, color="black")
compar_tab_14_17_20 = energy_burden_SR_LR_table(new_betas_df_energy_EB, 
                                       apt_yer_1 =14, 
                              apt_yer_2 =17,apt_yer_3 = 20, 
                              energy_burden_LR_scen ="Monthly_EB_Median_RCP85_LR",
                              energy_burden_SR_scen ="Monthly_EB_Median_RCP85_SR",
                              burden_threshold= 6)
compar_tab_14_17_20.columns = ["Long_Run", "Short_Run"]
compar_tab_14_17_20.loc[:,"Difference"]= compar_tab_14_17_20.loc[:,"Short_Run"].subtract(compar_tab_14_17_20.loc[:,"Long_Run"])

compar_tab_new_19_22_25 = energy_burden_SR_LR_table(new_betas_df_energy_EB, 
                                       apt_yer_1 =19, 
                              apt_yer_2 =22,apt_yer_3 = 25, 
                              energy_burden_LR_scen ="Monthly_EB_Median_RCP85_LR",
                              energy_burden_SR_scen ="Monthly_EB_Median_RCP85_SR",
                              burden_threshold= 6)
compar_tab_new_19_22_25.columns = ["Long_Run", "Short_Run"]
compar_tab_new_19_22_25.loc[:,"Difference"]= compar_tab_new_19_22_25.loc[:,"Short_Run"].subtract(compar_tab_new_19_22_25.loc[:,"Long_Run"])

compar_tab_new_40s = energy_burden_SR_LR_table(new_betas_df_energy_EB, 
                                       apt_yer_1 =40, 
                              apt_yer_2 =43,apt_yer_3 = 47, 
                              energy_burden_LR_scen ="Monthly_EB_Median_RCP85_LR",
                              energy_burden_SR_scen ="Monthly_EB_Median_RCP85_SR",
                              burden_threshold= 6)

compar_tab_new_40s.columns = ["Long_Run", "Short_Run"]
compar_tab_new_40s.loc[:,"Difference"]= compar_tab_new_40s.loc[:,"Short_Run"].subtract(compar_tab_new_40s.loc[:,"Long_Run"])

fig,axs = plt.subplots(dpi= 1100)
sns.barplot(data = compar_tab_new_40s.reset_index(),
            y="index", x="Difference", color="orange", label = "SEER 40,43,47",hatch="\\")
sns.barplot(data = compar_tab.reset_index(),
            y="index", x="Difference", color="red",label =  "SEER 24,27,30")
sns.barplot(data = compar_tab_new_19_22_25.reset_index(),
            y="index", x="Difference", color="steelblue", label = "SEER 19,22,25",
            hatch="//")
sns.barplot(data = compar_tab_14_17_20.reset_index(),
            y="index", x="Difference", color="gray",label =  "SEER 14,17,20", 
            )
sns.despine(top= True, right=True)
axs.set_xlim(0,25)
axs.axvline(0,linewidth=5, color="black")

axs.set_xlabel("Count of Households")
plt.legend(frameon= False)
# %%

new_betas_df_energy_EB.loc[
    ((new_betas_df_energy_EB.Year >= 2025) & ((new_betas_df_energy_EB.Year <= 2044))), "Term"] = "Near_Term"
new_betas_df_energy_EB.loc[
    ((new_betas_df_energy_EB.Year >= 2045) & ((new_betas_df_energy_EB.Year <= 2065))), "Term"] = "Mid_Century"
new_betas_df_energy_EB.loc[
    ((new_betas_df_energy_EB.SEER_labels == 15)), "Eff_Scen"] = "Seer 14"
new_betas_df_energy_EB.loc[
    ((new_betas_df_energy_EB.SEER_labels == 20)), "Eff_Scen"] = "Seer 20"
new_betas_df_energy_EB.loc[
    ((new_betas_df_energy_EB.SEER_labels == 25)), "Eff_Scen"] = "Seer 25"
new_betas_df_energy_EB.loc[:, "Decade"] = (
    np.floor(new_betas_df_energy_EB.loc[:, "Year"].astype(int)/10)*10).astype(int)

plot_data_df = pd.concat([new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2020) &
 (new_betas_df_energy_EB["Year"]<= 2040) & 
 (new_betas_df_energy_EB["Cal_list"]==15))],
new_betas_df_energy_EB[(
(new_betas_df_energy_EB["Year"]>=2041) &
 (new_betas_df_energy_EB["Year"]<= 2061) & 
 (new_betas_df_energy_EB["Cal_list"]==18))],
new_betas_df_energy_EB[(
(new_betas_df_energy_EB["Year"]>=2062) &
 (new_betas_df_energy_EB["Year"]<= 2070) & 
 (new_betas_df_energy_EB["Cal_list"]==21))]],
axis=0).groupby(["acct","Year_s","models","Cal_list"]).mean()[
    ['Monthly_EB_Median_RCP85_LR', 'Monthly_EB_Median_RCP85_SR']]
    
plot_data_df_21 = pd.concat([new_betas_df_energy_EB[(
    (new_betas_df_energy_EB["Year"]>=2020) &
 (new_betas_df_energy_EB["Year"]<= 2040) & 
 (new_betas_df_energy_EB["Cal_list"]==22))],
new_betas_df_energy_EB[(
(new_betas_df_energy_EB["Year"]>=2041) &
 (new_betas_df_energy_EB["Year"]<= 2061) & 
 (new_betas_df_energy_EB["Cal_list"]==25))],
new_betas_df_energy_EB[(
(new_betas_df_energy_EB["Year"]>=2062) &
 (new_betas_df_energy_EB["Year"]<= 2070) & 
 (new_betas_df_energy_EB["Cal_list"]==28))]],
axis=0).groupby(["acct","Year_s","models","Cal_list"]).mean()[
    ['Monthly_EB_Median_RCP85_LR', 'Monthly_EB_Median_RCP85_SR']]
plot_data_df_mod=add_demo_data_ele_sim(plot_data_df.reset_index(), past_data_annual.reset_index())
plot_data_df_mod_21=add_demo_data_ele_sim(plot_data_df_21.reset_index(),
                                          past_data_annual.reset_index())


plot_data_df_mod.groupby([])

plot_data_burden = plot_data_df_mod.melt(id_vars=['IG_num',"Cal_list", 'Race', "Year_s",'Age',"models","acct"],
     value_vars=['Monthly_EB_Median_RCP85_LR', 'Monthly_EB_Median_RCP85_SR'],
     var_name = "Scenarios",
      value_name="Median Energy Burden")

plot_data_burden_21 = plot_data_df_mod_21.melt(id_vars=['IG_num',"Cal_list", 'Race', "Year_s",'Age',"models","acct"],
     value_vars=['Monthly_EB_Median_RCP85_LR', 'Monthly_EB_Median_RCP85_SR'],
     var_name = "Scenarios",
      value_name="Median Energy Burden")

plot_data_burden.loc[:,"IG_num"]= plot_data_burden.loc[:,"IG_num"].astype("category")
plot_data_burden.replace("Monthly_EB_Median_RCP85_LR","Long_Run", inplace= True)
plot_data_burden.replace("Monthly_EB_Median_RCP85_SR","Short_Run", inplace= True)
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]>6,"High_Energy_burden"]= 1
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]<=6,"High_Energy_burden"]= 0
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]>10,"Severe_Energy_burden"]= 1
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]<=10,"Severe_Energy_burden"]= 0
plot_data_burden.loc[plot_data_burden.acct.isin(accts_benefiting_85),"Benefits"]= "Benefits"
plot_data_burden.loc[plot_data_burden.Benefits!="Benefits","Benefits"] ="Stays the Same"

plot_data_burden.loc[:,"IG_num"]= plot_data_burden.loc[:,"IG_num"].astype("category")
plot_data_burden.replace("Monthly_EB_Median_RCP85_LR","Long_Run", inplace= True)
plot_data_burden.replace("Monthly_EB_Median_RCP85_SR","Short_Run", inplace= True)
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]>6,"High_Energy_burden"]= 1
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]<=6,"High_Energy_burden"]= 0
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]>10,"Severe_Energy_burden"]= 1
plot_data_burden.loc[plot_data_burden["Median Energy Burden"]<=10,"Severe_Energy_burden"]= 0
plot_data_burden.loc[plot_data_burden.acct.isin(accts_benefiting_85),"Benefits"]= "Benefits"
plot_data_burden.loc[plot_data_burden.Benefits!="Benefits","Benefits"] ="Stays the Same"

plot_data_burden_21.loc[:,"IG_num"]= plot_data_burden_21.loc[:,"IG_num"].astype("category")
plot_data_burden_21.replace("Monthly_EB_Median_RCP85_LR","Long_Run", inplace= True)
plot_data_burden_21.replace("Monthly_EB_Median_RCP85_SR","Short_Run", inplace= True)
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]>6,"High_Energy_burden"]= 1
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]<=6,"High_Energy_burden"]= 0
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]>10,"Severe_Energy_burden"]= 1
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]<=10,"Severe_Energy_burden"]= 0
plot_data_burden_21.loc[plot_data_burden_21.acct.isin(accts_benefiting_85),"Benefits"]= "Benefits"
plot_data_burden_21.loc[plot_data_burden_21.Benefits!="Benefits","Benefits"] ="Stays the Same"

plot_data_burden_21.loc[:,"IG_num"]= plot_data_burden_21.loc[:,"IG_num"].astype("category")
plot_data_burden_21.replace("Monthly_EB_Median_RCP85_LR","Long_Run", inplace= True)
plot_data_burden_21.replace("Monthly_EB_Median_RCP85_SR","Short_Run", inplace= True)
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]>6,"High_Energy_burden"]= 1
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]<=6,"High_Energy_burden"]= 0
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]>10,"Severe_Energy_burden"]= 1
plot_data_burden_21.loc[plot_data_burden_21["Median Energy Burden"]<=10,"Severe_Energy_burden"]= 0
plot_data_burden_21.loc[plot_data_burden_21.acct.isin(accts_benefiting_85),"Benefits"]= "Benefits"
plot_data_burden_21.loc[plot_data_burden_21.Benefits!="Benefits","Benefits"] ="Stays the Same"




plot_test = plot_data_burden.groupby([
    "acct","models","Year_s","Scenarios","IG_num","Benefits"]).sum()[
        "High_Energy_burden"].sum(level=["models",
                                         "Scenarios","Year_s","IG_num","Benefits"]).reset_index()

plot_test_21 = plot_data_burden_21.groupby([
    "acct","models","Year_s","Scenarios","IG_num","Benefits"]).sum()[
        "High_Energy_burden"].sum(level=["models",
                                         "Scenarios","Year_s","IG_num","Benefits"]).reset_index()

plot_test.loc[:,"Decade"] = (
    np.floor(plot_test.loc[:,"Year_s"].astype(int)/10)*10).astype(int).astype('category')
plot_test.loc[:,"IG_num"] = plot_test.loc[:,"IG_num"].astype('category')

plot_data_burden.acct.nunique()

counts_hh_IG_21 = plot_data_burden_21[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","IG_num","Scenarios"]).nunique()["acct"]
counts_hh_Age_21 = plot_data_burden_21[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","Age","Scenarios"]).nunique()["acct"]
counts_hh_Race_21 = plot_data_burden_21[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","Race","Scenarios"]).nunique()["acct"]       

fig,axs= plt.subplots(3,1,dpi=1100,figsize=(9,15),
                      sharex=True)             
sns.barplot(ax=axs[0],data=counts_hh_IG_21.reset_index(),
            y="IG_num", x="acct",
            units="models", hue="Scenarios")
sns.barplot(ax=axs[1],data=counts_hh_Age_21.reset_index(), y="Age", x="acct",
            units="models", hue="Scenarios")
sns.barplot(ax=axs[2],data=counts_hh_Race_21.reset_index(), y="Race", x="acct",
            units="models", hue="Scenarios")

counts_hh_IG = plot_data_burden[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","IG_num","Scenarios"]).nunique()["acct"]
counts_hh_Age = plot_data_burden[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","Age","Scenarios"]).nunique()["acct"]
counts_hh_Race = plot_data_burden[
    plot_data_burden["Median Energy Burden"]>6].groupby(
        ["Year_s","models","Race","Scenarios"]).nunique()["acct"]       

fig,axs= plt.subplots(3,1,dpi=1100,figsize=(9,15),
                      sharex=True)             
sns.barplot(ax=axs[0],data=counts_hh_IG.reset_index(),
            y="IG_num", x="acct",
            units="models", hue="Scenarios")
sns.barplot(ax=axs[1],data=counts_hh_Age.reset_index(), y="Age", x="acct",
            units="models", hue="Scenarios")
sns.barplot(ax=axs[2],data=counts_hh_Race.reset_index(), y="Race", x="acct",
            units="models", hue="Scenarios")


ax = sns.FacetGrid(data=plot_data_burden[plot_data_burden.acct.isin(accts_benefiting_85)], col='IG_num',
                   col_wrap=2, height=3.5, aspect=1.5)
ax = sns.FacetGrid(data=plot_data_burden, col='IG_num',
                   col_wrap=2, height=3.5, aspect=1.5)
ax.map_dataframe(sns.pointplot, x="Year_s", 
                 y="High_Energy_burden" , hue="Scenarios",
                 estimator=np.sum, units= "acct")
ax.set_titles("{col_name}")
ax.set_xlabels("")

#plt.legend(loc="upper left")
ax.add_legend()
ax.refline(y=6, color="red", ls="--")
ax.refline(y=10,color="red", ls="--")
sns.move_legend(ax, "upper left",
                bbox_to_anchor=(.55, .30), frameon=False, 
                fontsize=15, 
                labels= ["Efficiency Changes SEER: 15,18,19",
                         "No SEER Changes","High EB", "Severe EB"])
ax = sns.FacetGrid(data=plot_data_burden[plot_data_burden.acct.isin(accts_benefiting_85)],
                   col='Age',
                   col_wrap=3, height=2.5, aspect=1.5)
ax = sns.FacetGrid(data=plot_data_burden,
                   col='Age',
                   col_wrap=3, height=2.5, aspect=1.5)
ax.map_dataframe(sns.lineplot, x="Year_s", 
                 y="Median Energy Burden" , hue="Scenarios",
                 estimator=np.median)
ax.set_titles("{col_name}")
ax.set_xlabels("")

#plt.legend(loc="upper left")
ax.add_legend()
ax.refline(y=6, color="red", ls="--", label= "High")
ax.refline(y=10,color="red", ls="--", label = "Severe")
sns.move_legend(ax, "upper left",
                bbox_to_anchor=(.55, .30), frameon=False, 
                fontsize=15, 
                labels= ["Efficiency Changes SEER: 15,18,19",
                         "No SEER Changes","High EB", "Severe EB"])

#Test the number of urdend
ax = sns.FacetGrid(data=plot_test,
                   hue="Scenarios",
                   col="Decade", height=4, aspect=1.2)
ax.map_dataframe(sns.barplot, 
                 y="High_Energy_burden", x="Decade",
                 estimator=np.mean,units="models"
 )
# ax.set_titles("{col_name}")
ax.set_xlabels("")

#plt.legend(loc="upper left")
ax.add_legend()
sns.catplot(data=plot_test,
                   col='IG_num',x="Scenarios",col_wrap=3,
                    height=3.5, aspect=1.2,
                 y="High_Energy_burden", hue="Benefits",
                 estimator=np.median,units="Year_s",
  kind='bar')
ax.refline(y=6, color="red", ls="--", label= "High")
ax.refline(y=10,color="red", ls="--", label = "Severe")
sns.move_legend(ax, "upper left",
                bbox_to_anchor=(.55, .30), frameon=False, 
                fontsize=15, 
                labels= ["Efficiency Changes SEER: 15,18,19",
                         "No SEER Changes","High EB", "Severe EB"])


g = sns.FacetGrid(data=plot_data, col="Scenarios",row= "Race",
                   height=3, aspect=1)
g.map_dataframe(sns.boxplot, x="Median Energy Burden",
                 y="models",   whis=[10, 90],      
                 flierprops = dict(marker='o', markersize=0.5, color = "gray"),
 )

g.refline(x=6)
g.refline(x=10)

g.add_legend()
g = sns.FacetGrid(data=plot_data.groupby("acct").median(), col= "Race",
                   col_wrap=3,height=3, aspect=1)
g.map_dataframe(sns.boxplot, x="Median Energy Burden",
                 hue="Scenarios", y="models",  whis=[10, 90],      
                 flierprops = dict(marker='o', markersize=0.5, color = "gray"),
 )



g.add_legend()

g = sns.FacetGrid(data=plot_data, col= "Race",
                   col_wrap=3,height=3, aspect=1)
g.map_dataframe(sns.pointplot, x="Median Energy Burden",
                 y="models", hue="Scenarios",estimator= np.median, 
                 joint=False, units= "acct" )

g.refline(x=6)
g.refline(x=10)

g.add_legend()

g = sns.Facetgr(new_betas_df_energy_EB[new_betas_df_energy_EB.SEER_labels == 20],
                col="Decade", hue="IG_num", col_wrap=2, kind="bar",
                height=3, aspect=1, estimator=np.median)

g = sns.catplot(data=new_betas_df_energy_EB[(((new_betas_df_energy_EB.SEER_labels == 15) |
                                              (new_betas_df_energy_EB.SEER_labels == 18) | (new_betas_df_energy_EB.SEER_labels == 21) |
                                              (new_betas_df_energy_EB.SEER_labels == 24) | (new_betas_df_energy_EB.SEER_labels == 27)) & ~(
    (new_betas_df_energy_EB.Term.isnull()) & (
        new_betas_df_energy_EB.Eff_Scen.isnull())
))], y="SEER_labels", x="Monthly_EB_Median_RCP85", hue="Term",  kind="bar",
    height=4, aspect=.6, col="Race", col_wrap=3, units="models")

g = sns.catplot(data=new_betas_df_energy_EB[~(
    (new_betas_df_energy_EB.Term.isnull()) & (
        new_betas_df_energy_EB.Eff_Scen.isnull())
)], x="SEER_labels", y="Monthly_EB_Median_RCP85", hue="Term",  kind="bar",
    height=4, aspect=.6, col="IG_num", col_wrap=3, units="models")

g.refline(x=6)
g.refline(x=10)

g.add_legend()

sns.pointplot(data=new_betas_df_energy_EB[new_betas_df_energy_EB.SEER_labels == 20], estimator=np.median,
              x="models", y="Monthly_EB_Median_RCP85", seed=32,
              hue="IG_num", join=False, ci=95, n_boot=5000)
share_pop_EB = pd.DataFrame(pd.concat([
    new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 6)].groupby(["Decade", "IG_num", "SEER_labels"])["acct"].nunique().divide(
        new_betas_df_energy_EB.groupby(["Decade", "IG_num", "SEER_labels"])["acct"].nunique()).multiply(100),
    new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 6)].groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique().divide(
        new_betas_df_energy_EB.groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique()).multiply(100),
    new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 6)].groupby(["Decade", "Age", "SEER_labels"])["acct"].nunique().divide(
            new_betas_df_energy_EB.groupby(["Decade", "Age", "SEER_labels"])["acct"].nunique()).multiply(100)], axis=0)
)
share_pop_EB["EB"] = pd.concat([
    new_betas_df_energy_EB.groupby(["Decade", "IG_num", "SEER_labels"])[
        "Monthly_EB_Median_RCP85"].median(),
    new_betas_df_energy_EB.groupby(["Decade", "Race", "SEER_labels"])[
        "Monthly_EB_Median_RCP85"].median(),
    new_betas_df_energy_EB.groupby(["Decade", "Age", "SEER_labels"])[
        "Monthly_EB_Median_RCP85"].median()
], axis=0)

g = sns.FacetGrid(share_pop_EB.reset_index()[
    share_pop_EB.reset_index().IG_num.isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ])], col="Decade", col_wrap=3)
g.map_dataframe(sns.scatterplot, x="EB", y="acct", hue="IG_num")
g.add_legend()
g = sns.FacetGrid(share_pop_EB.reset_index()[
    share_pop_EB.reset_index().IG_num.isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ])], col="Decade", col_wrap=3)
g.map_dataframe(sns.scatterplot, x="EB", y="acct", hue="IG_num")
g.add_legend()

sns.boxplot(data=new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 6)].groupby(["Decade", "IG_num", "SEER_labels"])["acct"].nunique().divide(
            new_betas_df_energy_EB.groupby(["Decade", "IG_num", "SEER_labels"])["acct"].nunique()).reset_index(),
            x="acct", y="IG_num")
sns.boxplot(data=new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 6)].groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique().divide(
            new_betas_df_energy_EB.groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique()).reset_index(),
            x="acct", y="Race")
sns.boxplot(data=new_betas_df_energy_EB[(new_betas_df_energy_EB.Monthly_EB_Median_RCP85 > 10)].groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique().divide(
            new_betas_df_energy_EB.groupby(["Decade", "Race", "SEER_labels"])["acct"].nunique()).reset_index(),
            x="acct", y="Race")

sns.boxplot(data=summertime_EB[(summertime_EB.Monthly_EB_Median > 6)].groupby(["Decade", "IG_num"])["acct"].nunique().divide(
            summertime_EB.groupby(["Decade", "IG_num"])["acct"].nunique()).reset_index(),
            y="acct", x="IG_num")
