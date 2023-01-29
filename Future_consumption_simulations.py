# -*- coding: utf-8 -*-
"""
This script saves creates climate electricity simulated consumption
@author: Andrew Jones
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions_and_labels import labels,filtered_hh 
#------------------------------------------------------------------------------
#%% Inputs
path = 'Insert path location of the folder to save the data in'

annual_cooling_points = pd.read_csv(path + '/Data/Baseline/Individual_Cooling_Balance_points.csv').set_index("BILACCT_K")
annual_heating_points = pd.read_csv(path + '/Data/Baseline/Individual_Heating_Balance_points.csv').set_index("BILACCT_K")
intercept_coef = pd.read_excel(path + '/Data/Baseline/2015_2019_intercept_coefficients.xlsx',header=0,
                               index_col=(0))
CDD_coeff = pd.read_excel(path + '/Data/Baseline/2015_2019_CDD_coefficients.xlsx',
                         header=0,
                          index_col=(0))
HDD_coeff = pd.read_excel(path + '/Data/Baseline/2015_2019_HDD_coefficients.xlsx',
                         header=0,
                          index_col=(0))
# panel data is the daily electricity consumption for the households
panel_data = pd.read_csv(path+'/Data/Daily_consumption.csv', parse_dates=[0],
                       index_col=[0])
exogen= pd.read_csv(path+'/Data/exogen_data_mod.csv', parse_dates=[0],
                       index_col=[0])

models_2015_2019= pd.read_pickle(path +'/Data/Baseline/2015_2019_regression_models.pkl')

#Have to rename the columns because of an initial mistake in naming 


# five_pt_1516_best_model_parameters = pd.read_excel(path+r'/Data/Baseline/2015_2016/2015_2016_best_paramters.xlsx')
# five_pt_1617_best_model_parameters = pd.read_excel(path+r'/Data/Baseline/2016_2017/2016_2017_best_paramters.xlsx')
five_pt_1718_best_model_parameters = pd.read_excel(path+r'/Data/Baseline/2017_2018/2017_2018_best_paramters.xlsx',
                                              index_col=[0]).dropna(axis=0)
# five_pt_1819_best_model_parameters = pd.read_excel(path+r'/Data/Baseline/2018_2019/2018_2019_best_paramters.xlsx')



temps = exogen["temp_avg"].groupby(exogen["temp_avg"].index.date).mean()

temps.index = pd.to_datetime(temps.index, format= "%Y/%m/%d")
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
year_labels =  ["15/16", "16/17", "17/18", "18/19"]

#%% Identify vulnerable households 
survey_all = pd.read_stata('Insert the file location of the survey data here')

survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7,4,5,6,2,1,8,3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"]!=""].unique()):
    survey_all.loc[survey_all["VINCOME"]==ig_num,["VINCOME_new"]] = int(ig_unsorted[idxs])
electric_households = survey_all["BILACCT_K"][~((survey_all.VHEATEQUIP =="Separate gas furnace") |
                                                    (survey_all.VHEATEQUIP =='Gas furnace packaged with AC unit (sometimes called a gas pa)'))].values

#%% Income Group Tables 
survey_all_IG_tabl = survey_all.copy(deep=True).set_index("BILACCT_K")
start_date_model ="05-01-2017"
end_date_model = "04-30-2018"
start_date = "05-01-2025"
end_date = "04-30-2026"
RCP_45_input = LOCA_future_RCP45_PX_runs
RCP_85_input =LOCA_future_RCP85_PX_runs
model_input_year = year_labels[2]
#%% Runs the climate projections with a trained year's model 
def climate_projections(RCP_45_input, RCP_85_input,
                        model_input_year,five_pt_1718_best_model_parameters,
                        start_date,end_date,
                        start_date_model,end_date_model):
    """Outputs the daily kwh 
    Model inputs: 
    RCP_45_input    
    RCP_85_input
    model_input_year: This input identifies the trained fixed effect model that the climodels_2014_2019.loc[mate-driven electricity consumption are derived from 
    climate_model_input_year: This input is the climate model of interest 
        """
    #Average temperatures for the time frame
    avg_daily_temp = exogen["temp_avg"].groupby(exogen.index.date).mean()
    avg_daily_temp.index = pd.to_datetime(avg_daily_temp.index)
    avg_daily_temp = avg_daily_temp.loc[start_date_model:end_date_model]
    exogen_year = exogen.loc[start_date_model:end_date_model]

    #Dataframe of the balance points
    degree_days_cp = pd.concat(
        [annual_cooling_points[model_input_year], 
         annual_heating_points[model_input_year]
                                ], axis=1).dropna(axis=0)
    degree_days_cp.columns = ["CCP", "HCP"]
    #Empty dataframe that will store final results for each household's kWh
    simulations_df = []
   
    #Loops to predict household-level consumption
    for idx,house in enumerate(degree_days_cp[
            degree_days_cp.index.isin(models_2015_2019.dropna().index)].index.values):
        df = pd.DataFrame()
        #Creates dataframe with independent variables for the regression model 
        hh_exo_prediction_45 = exogen_year.loc[
            (exogen_year.BILACCT_K==house),:].drop(axis=1,
            labels = ["temp_avg", 
                      "BILACCT_K"])
        
        #Identifies the dates that the household does not have data for 
        missing_date = pd.date_range(start = start_date_model,
                                     end = end_date_model)
        
        missing_date = missing_date[~missing_date.isin(
            hh_exo_prediction_45.index)]
        
        #For the house that have more than 365 days, they are average by
        #their date (this should not be a problem if the dates don't repeat)
        if hh_exo_prediction_45.holiday.count()>365:
            hh_exo_prediction_45= hh_exo_prediction_45.groupby(hh_exo_prediction_45.index.date).mean()
            hh_exo_prediction_45.index = pd.to_datetime(hh_exo_prediction_45.index)
            
        hh_exo_prediction_45["HDD"]= np.nan
        hh_exo_prediction_45["CDD"]= np.nan
        hh_exo_prediction_cost = hh_exo_prediction_45["elec_cost"]
        hh_exo_prediction_45.loc[:,"intercept"] = 1
        hh_exo_prediction_85 = hh_exo_prediction_45.copy(deep=True)


        #Calculates the HDD for each of the models                                                                     
        for idx, run in enumerate(RCP_45_input.columns):
            
            #identifies the CDD slope for a household 
            hh_CDD_slope = CDD_coeff.loc[house,model_input_year]
                        
            Heat_45 = (degree_days_cp.loc[house,"HCP"]-RCP_45_input.loc[
                start_date:end_date, run]).clip(lower=0)
            Heat_85 = (degree_days_cp.loc[house,"HCP"]-RCP_85_input.loc[
                start_date:end_date, run]).clip(lower=0)
            
            #Determines if it is a leap year and then drops day 60 February 29
            if Heat_45.count() == 366:
                Heat_45 = Heat_45[Heat_45.index.dayofyear != 60]
                Heat_85 = Heat_85[Heat_85.index.dayofyear != 60]
                
            #If there is a missing date in the independent varaible 
            # dataframe, then it is removed
            if missing_date.empty==False:
                Heat_45= Heat_45[~(Heat_45.index.strftime('%m/%d').isin(
                    missing_date.strftime('%m/%d')))]
                Heat_85= Heat_85[~(Heat_85.index.strftime('%m/%d').isin(
                    missing_date.strftime('%m/%d')))]                                                       
            
            hh_exo_prediction_45["HDD"] = Heat_45.values 
            hh_exo_prediction_85["HDD"] = Heat_85.values
    
            #Calculates the CDD for each of the models  
            Cool_45 = (RCP_45_input.loc[start_date:end_date, run].subtract(
                degree_days_cp.loc[house,"CCP"])).clip(lower=0)
            Cool_85 = (RCP_85_input.loc[start_date:end_date, run].subtract(
                degree_days_cp.loc[house,"CCP"])).clip(lower =0)
            
            #Determines if it is a leap year and then drops day 60 February 29
            if Cool_45.count() == 366:
                Cool_45 = Cool_45[Cool_45.index.dayofyear != 60]
                Cool_85 = Cool_85[Cool_85.index.dayofyear != 60]
                
            #If there is a missing date in the exogenous dataframe, then it is removed
            if missing_date.empty==False:
                Cool_45= Cool_45[~(Cool_45.index.strftime('%m/%d').isin(
                    missing_date.strftime('%m/%d')))]
                Cool_85= Cool_85[~(Cool_85.index.strftime('%m/%d').isin(
                    missing_date.strftime('%m/%d')))]

            #Uses the temperatures that have the same day of the year (n=365) 
            #to ensure that the only thing that change are temperature
            hh_exo_prediction_45["CDD"] = Cool_45.values
            hh_exo_prediction_85["CDD"] = Cool_85.values
            
            #Finds the days where the household uses cooling
            #when the consumption is greater than zero
            cool_dates_45 = hh_exo_prediction_45["CDD"][
                hh_exo_prediction_45["CDD"]>0].index
            cool_dates_85 = hh_exo_prediction_85["CDD"][
                hh_exo_prediction_85["CDD"]>0].index
            
            #Identifies the fixed effects that needs to be dropped
            hh_drop_fixed_effects = five_pt_1718_best_model_parameters.loc[
                house,["day","month"]]
            
            #Predict the consumption with RCP 4.5 
            climate_simualtions_45 = models_2015_2019.loc[
            house,model_input_year].predict(
            hh_exo_prediction_45.drop(axis=1,labels = hh_drop_fixed_effects))
            
            #Predict the consumption with  RCP 8.5
            climate_simualtions_85 = models_2015_2019.loc[
            house,model_input_year].predict(
            hh_exo_prediction_85.drop(axis=1,labels = hh_drop_fixed_effects))

            #Calculates the total cost of kWh 
            df = pd.concat([climate_simualtions_45,climate_simualtions_85],axis=1)
            df.columns = ["kwh_RCP4.5", "kwh_RCP8.5"]
            df.index.columns = ["Date"]
            df["model"] = run
            df["acct"] = house
            df["cost_4.5"] = hh_exo_prediction_cost*climate_simualtions_45
            df["cost_8.5"] = hh_exo_prediction_cost*climate_simualtions_85
    
            #Finds the cooling load
            df["cool_kwh_RCP4.5"] = Cool_45.values *hh_CDD_slope
            df["cool_kwh_RCP8.5"] = Cool_85.values *hh_CDD_slope
            
            #Sets all days whose temperature is not greater than balance point to zero
            df.loc[~(climate_simualtions_45.index.dayofyear.isin(
                    cool_dates_45.dayofyear)), "cool_kwh_RCP4.5"]=0
            
            df.loc[~(climate_simualtions_85.index.dayofyear.isin(
                    cool_dates_85.dayofyear)), "cool_kwh_RCP8.5"]=0
            
            #Calculates the total cost of kWh 
            df["cool_cost_RCP4.5"] = hh_exo_prediction_cost*df["cool_kwh_RCP4.5"]
            df["cool_cost_RCP8.5"] = hh_exo_prediction_cost*df["cool_kwh_RCP8.5"]
            
            #Copys the future dates into the dataframe
            df["RCP_dates"] = Cool_45.index
            df = df.reset_index()
            df.rename(columns = {'index':"Date"}, inplace= True)

            simulations_df.append(df)
            
    simulations_data = pd.concat(simulations_df,axis=0)
        
    return(simulations_data)

climate_simualtions_historical_1718 = climate_projections(
    LOCA_historical_RCP45_PX_runs,LOCA_historical_RCP85_PX_runs, 
    year_labels[2],five_pt_1718_best_model_parameters,
    "05-01-2017","04-30-2018",
    "05-01-2017","04-30-2018")

climate_simualtions_historical_1718.to_csv(path + r'/Data/Consumption_Simulations/historical.csv')

climate_simualtions_historical_1718 = pd.read_csv(path + '/Data/Consumption_Simulations/historical.csv', 
                                                      date_parser=('date_s'),low_memory=False)

#%% Creates the data frame with the future consumption 2030-2070
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

def climate_projection_data():
    """Write the code description here"""
    #Runs the fuction for the simulated temperatures in the 2017-2018 model 
    list_of_data_annual = []
    list_of_data_month = []
    
    for j in range(len(climate_years_range)):
        tot_day_projection = climate_projections(
             LOCA_future_RCP45_PX_runs, 
             LOCA_future_RCP85_PX_runs, 
             year_labels[2],five_pt_1718_best_model_parameters,
             climate_years_range.loc[j,"start_date"],
             climate_years_range.loc[j,"end_date"],
             "05-01-2017","04-30-2018") 
        print("Finished",climate_years_range.loc[j,"start_date"],"-"\
              ,climate_years_range.loc[j,"end_date"])
        
        #Dataframe with a year's worth of electricity simulations
        hh_data=  tot_day_projection.set_index(["acct","model","RCP_dates"])
 
        annual_acct = hh_data.groupby(["acct","model"]).sum()
        
        annual_acct.loc[:,"Year"] = climate_years_range.loc[
            j,"start_date"].split("-")[0]+"/" +\
            climate_years_range.loc[j,"end_date"].split("-")[0]   
        annual_acct.loc[:,"Decade"] = int(climate_years_range_dec["Decade"][j])
        
        month_acct = hh_data.groupby(["acct","model",hh_data.index.get_level_values("RCP_dates").month
                                      ]).sum()
        month_acct.loc[:,"Year"] = climate_years_range.loc[
            j,"start_date"].split("-")[0]+"/" +\
            climate_years_range.loc[j,"end_date"].split("-")[0]   
        month_acct.loc[:,"Decade"] = int(climate_years_range_dec["Decade"][j]) 
       
        annual_acct.reset_index(inplace=True)
        month_acct.reset_index(inplace=True)
        
        list_of_data_annual.append(annual_acct)
        list_of_data_month.append(month_acct)
    future_data_annual = pd.concat(list_of_data_annual, axis=0)
    future_data_monthly = pd.concat(list_of_data_month, axis=0)
    future_data_annual.to_csv(path + r'/Data/Consumption_Simulations/Annual_estimates.csv')
    future_data_monthly.to_csv(path + r'/Data/Consumption_Simulations/Monthly_estimates.csv')
    return (future_data_annual,future_data_monthly)
[future_annual_data, future_monthly_data] = climate_projection_data()

future_annual_data = pd.read_csv(path + '/Data/Consumption_Simulations/Annual_estimates.csv')
future_monthly_data = pd.read_csv(path + '/Data/Consumption_Simulations/Monthly_estimates.csv')
#%% Calculates the percentage change for annual

#Creates a new df with the the consumption summed by new index
future_data_annual= future_annual_data.groupby(
    ["acct", "model","Year","Decade"]).sum()

past_data_annual = climate_simualtions_historical_1718.groupby(["acct", "model"]
                                                               ).sum()

#Joins the two df and copies the past data into the matching index (repeats past values for each matching
# future year)
joined_annual_data = future_data_annual.join(past_data_annual, on=["acct", "model"],
                              rsuffix= "_past")
past_data_annual_new = joined_annual_data.loc[:,['kwh_RCP4.5_past', 'kwh_RCP8.5_past',
                                              'cost_4.5_past', 'cost_8.5_past', 'cool_kwh_RCP4.5_past',
                                              'cool_kwh_RCP8.5_past','cool_cost_RCP4.5_past',
                                              'cool_cost_RCP8.5_past']]
past_data_annual_new.columns = future_data_annual.columns

#Calculates the percentage change
annual_per_change = future_data_annual.div(past_data_annual_new).subtract(1).multiply(100)
annual_per_change.reset_index(inplace= True)
#Calculates the relative consumption changes
annual_relative_change = future_data_annual.subtract(past_data_annual_new)
past_data_annual.reset_index(inplace= True)
def add_demo_data_ele_sim(annual_per_change):
    """This function uses the axis of the dataframe indexed with acct, model, years, and
    decade information to add income, race, and age data"""
    
    for i,house in enumerate(past_data_annual.acct.unique()):
        #Income 
        annual_per_change.loc[annual_per_change.acct== house,
                              "IG_num"] = survey_all["VINCOME_new"][survey_all[
            "BILACCT_K"] == house].values[0]     
        #Race
        annual_per_change.loc[annual_per_change.acct== house,
                              "Race"] = survey_all["VETHNIC"][survey_all[
            "BILACCT_K"] == house].values[0]
        #Age   
        annual_per_change.loc[annual_per_change.acct== house,
                              "Age"] = survey_all["VHH_AGECODE"][survey_all[
            "BILACCT_K"] == house].values[0]
        #AC Type   
        annual_per_change.loc[annual_per_change.acct== house,
                              "AC_Type"] = survey_all["VACTYPE"][survey_all[
            "BILACCT_K"] == house].values[0] 
        #Dwelling Size
        annual_per_change.loc[annual_per_change.acct== house,
                              "Sqft"] = survey_all["VSQFEET"][survey_all[
            "BILACCT_K"] == house].values[0]
        #Household Members
        annual_per_change.loc[annual_per_change.acct== house,
                              "Occupancy"] = survey_all["VHOUSEHOLD"][survey_all[
            "BILACCT_K"] == house].values[0]
        #AC Units
        annual_per_change.loc[annual_per_change.acct== house,
                              "ACUNITS"] = survey_all["VACUNITS"][survey_all[
            "BILACCT_K"] == house].values[0]
    annual_per_change.replace(
        {"65-74 yrs old":"65 yrs or older", 
        "75+ yrs old":"65 yrs or older"}, inplace= True)
    return(annual_per_change)


annual_per_change = add_demo_data_ele_sim(annual_per_change)
annual_relative_change = add_demo_data_ele_sim(annual_relative_change.reset_index())
future_annual_data.to_csv(path + r'/Data/Consumption_Simulations/Annual_estimates.csv')
future_monthly_data.to_csv(path + r'/Data/Consumption_Simulations/Monthly_estimates.csv')

#Saves the data
annual_per_change.to_csv(path + r'/Data/Consumption_Simulations/Annual_per_change_estimates.csv')
annual_relative_change.to_csv(path + r'/Data/Consumption_Simulations/Annual_relative_change_estimates.csv')
#%% 

