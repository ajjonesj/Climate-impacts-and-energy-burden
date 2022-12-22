# -*- coding: utf-8 -*-
"""
This script performs and save the temperature response regression
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'C:/Users/andre/Box/Andrew Jones_PhD Research/Climate change impacts on future electricity consumption and energy burden'

survey_all = pd.read_stata('C:/Users/andre/Box/RAPID COVID Residential Energy/Arizona_Full datasets/Arizona 2019-2020 dataset/RET_2017_part1.dta')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7,4,5,6,2,1,8,3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"]!=""].unique()):
    survey_all.loc[survey_all["VINCOME"]==ig_num,["VINCOME_new"]] = int(ig_unsorted[idxs])


def weather_data():
    humidity_data = pd.read_csv(path+'/Data/humdity_2014_2019_data.csv',
                                                   parse_dates=['DateTime'])
    humidity_data =humidity_data.set_index("DateTime").dropna()
    humidity_data = humidity_data.mean(axis=1)
    humidity_data.index = pd.to_datetime(humidity_data.index,format= "%m/%d/%Y" )
    
    climate_years_range_mods= pd.DataFrame(data=[
        pd.date_range(start= "05-01-2014",  periods = 5,
                      freq=pd.DateOffset(years=1)),
        pd.date_range(start= "04-30-2019",  periods = 5,
                      freq=pd.DateOffset(years=1))],
        ).T
    climate_years_range_mods.columns = ["start_date", "end_date"]
    
    #Replaces the inital temperature from Weather.com with readings from NOAA
    NOAA_data =  pd.read_csv(path + "/Data/temp_2014_2019_data.csv",  parse_dates=['DATE'])
    NOAA_data["TAVG"] = NOAA_data["TMAX"].add(NOAA_data["TMIN"]).divide(2)
    NOAA_data.set_index('DATE', inplace= True)
    NOAA_data2= NOAA_data.groupby(NOAA_data.index.date).mean()["TAVG"]
    return(humidity_data,NOAA_data2)

[humidity_data,NOAA_data2] = weather_data()

panel_data= pd.read_csv(path+'/Data/Daily_consumption.csv', parse_dates=[0],
                       index_col=[0])
exogen= pd.read_csv(path+'/Data/exogen_data_mod.csv', parse_dates=[0],
                       index_col=[0])

def panel_data_fun():
    """The function uses the data that Ali created to create a dataframe from
    May 2014 to March 2019""" 
    summer_2014_2015 = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2014_2015_s_daily.dta")
    spring_2014_2015 = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2014_2015_sp_daily.dta")
    winter_2014_2015 = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2014_2015_w_daily.dta").drop(axis=1,labels="index")
    data_2015_2016  = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2015_2016_daily.dta")
    data_2016_2017  = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2016_2017_daily.dta")
    data_2017_2018  = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2017_2018_daily.dta")
    data_2018_2019  = pd.read_stata("C:/Users/andre/Box/RAPID COVID Residential Energy/Results and coding files for Shuchen/2018_2019_daily.dta")

    panel_data = pd.concat([summer_2014_2015,spring_2014_2015,winter_2014_2015,
                        data_2015_2016,data_2016_2017,data_2017_2018,data_2018_2019
                        ],axis=0)
    panel_data["date_s"] = pd.to_datetime(panel_data["date_s"], format= "%Y/%m/%d")
    panel_data = panel_data.drop(axis=1,labels=["index","month","temp_avg_sq"])
    return(panel_data)
    
panel_data_all_full = panel_data_fun()
tot_hhs = panel_data_all_full.BILACCT_K.unique()

def locate_households_obs(panel_data_all_full):
    """This function creates a dataframe that list each account that has at
    least 360 observations """
    #Defines a new dataframe that only includes the daily electricirt consumption, 
    panel_data0 = panel_data_all_full.loc[:,["BILACCT_K","he_d","date_s"]].set_index(["date_s"])
    panel_data0= panel_data0.groupby([panel_data0.BILACCT_K,panel_data0.index])["he_d"].sum()
    test = pd.DataFrame(index =pd.date_range("2014-05-01","2019-04-30", freq='D'),
                        columns = panel_data0.index.get_level_values("BILACCT_K").unique()
                        )
    #This loop groups the households into into columns 
    for row,household in enumerate(test.columns):
        test.loc[panel_data0[panel_data0.index.get_level_values("BILACCT_K") ==household].index.get_level_values("date_s"),
                 household] = panel_data0[panel_data0.index.get_level_values("BILACCT_K") ==household].values
    #identifies households that have 365 days worth of observations
    full_year_data = test.groupby(test.index.year).count()
    full_year_data_tot = full_year_data.sum()
    full_year_data_tot.max()
    hh_with_more_than_half =full_year_data_tot[full_year_data_tot>913].index
      
    pan_data = test.loc[:,hh_with_more_than_half]

    return(pan_data)

#Uncomment when running the code from the beginning
panel_data = locate_households_obs(panel_data_all_full)

exogen = panel_data_all_full.drop(axis=1,labels=["summer","he_d","winter","weekend",
                                                   "RATE","summer_peak"]).set_index("date_s")
exogen = exogen[(exogen.BILACCT_K.isin(panel_data.columns))]
exogen = exogen.loc["2014-05-01":"2019-04-30"]
exogen["temp_avg"] =  NOAA_data2

#Finds missing electricity price and sets the price as the average for the day before and after the missing date
if exogen.elec_cost.isnull().any():
    idxs = exogen.reset_index()
    accts_with_missing_elec_cost = pd.DataFrame(idxs["BILACCT_K"][idxs.elec_cost.isnull()])
    for row,acct in enumerate(accts_with_missing_elec_cost.BILACCT_K.unique()):
        idx = accts_with_missing_elec_cost.index[0]
        #-1 is the electricity price columns
        household_data = exogen[exogen.BILACCT_K==acct].reset_index()
        missing_row = household_data[household_data.elec_cost.isnull()].index.values
        new_elec_price = (
        household_data.iloc[missing_row+1,-1].values[0]+\
            exogen.iloc[missing_row-1,-1].values[0])/2
        exogen.iloc[idx,-1] = new_elec_price

exogen.to_csv(path+'/Data/exogen_data_mod.csv')
panel_data.to_csv(path+ '/Data/Simulated_daily_consumption.csv')
panel_data = pd.read_csv(path+ '/Data/Daily_consumption.csv',parse_dates=[0],
                       index_col=[0])

#%%Code to find the five point regression
"""This section creates the functions to find the balance points for both
 cooling and heating. The optimal values are believed to lie within the ranges presented
 in the 'cooling_points' and 'heating_points' range """

#------------------------------------------------------------------------------
#Piece-wise linear regression
#------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold 
from sklearn.model_selection import cross_val_score 

panel_data_piece= panel_data.copy(deep= True)

def five_parameter_regression(start_date, end_date,folds, n_repeats,c_year):
    """Write the description here
    years include """
    #Creates an empty dataframe to store the 5-parameter regression with fixed effets
    #adjusted r_squared values for each 
    
    piecewise_model_r_square_adj = pd.DataFrame(
        index=panel_data_piece.columns,
        columns = ["R_squared_adj"])
    
    rMSE = pd.DataFrame(index=panel_data.columns,
                        columns = ["mod1","mod2","mod3", "mod4"])
    best_model_parameters = pd.DataFrame(index =panel_data_piece.columns, 
                                         columns = ["HBP","CBP","day","month",
                                                    "best_model"])
    #Creates an empty dataframe to store the model coeffieints from the 5-pt
    #with the fixed effects
    all_models_adjr2 = pd.DataFrame(index =panel_data_piece.columns, 
                                         columns = ["model1_adjr2",
                                                    "model2_adjr2",
                                                    "model3_adjr2",
                                                    "model4_adjr2"])
    piecewise_model_coefficient = pd.DataFrame(
        index=panel_data.columns, 
        columns = [exogen.drop(axis=1,labels=["temp_avg","BILACCT_K"]
                               ).columns])
    piecewise_model_coefficient["HDD"] = np.nan
    piecewise_model_coefficient["CDD"] = np.nan
    piecewise_model_coefficient["intercept"] = np.nan
    piecewise_model_p_values = piecewise_model_coefficient.copy(deep=True)
    five_pt_models = pd.DataFrame(index=panel_data_piece.columns,
                                  columns = ["14/15", "15/16","16/17", "17/18",
                                             "18/19"])
    five_pt_models1= five_pt_models.copy(deep=True)
    five_pt_models2= five_pt_models.copy(deep=True)
    five_pt_models3= five_pt_models.copy(deep=True)
    five_pt_models4= five_pt_models.copy(deep=True)

    panel_data_year = panel_data.loc[start_date:end_date]
    panel_data_year.columns = panel_data_year.columns.astype(int)
    piece_exogen = exogen.loc[start_date:end_date]

    piece_exogen["avg_rel_humidity"] = humidity_data.loc[start_date:end_date]
    predictions = pd.DataFrame(columns=panel_data_piece.columns,
                              index= panel_data_piece.index)
    hh_count = []
    
    # for i in :
    #     sleep(0.02)
    for idx,households in enumerate(panel_data_year.columns):
        hh_count.append(households) 
        
        mod_entries= []
        hh_exo = piece_exogen.loc[piece_exogen["BILACCT_K"]==households].set_index(
            piece_exogen[piece_exogen["BILACCT_K"]==int(households)].index)
        if hh_exo.empty:
            continue
        #Ensures that I do not repeat days 
        hh_exo = hh_exo.groupby(hh_exo.index.date).mean()
        
        hh_elec = panel_data_year.loc[:,households]
        hh_elec = hh_elec.groupby(hh_elec.index.date).mean()
        hh_elec.index = pd.to_datetime(hh_elec.index,format= "%Y/%m/%d")
        hh_exo.index = pd.to_datetime(hh_exo.index,format= "%Y/%m/%d")
        hh_elec = hh_elec[hh_elec.index.isin(hh_exo.index)]
        hh_exo = hh_exo[hh_exo.index.isin(hh_exo.index)]
        hh_exo["HDD"]= np.nan
        hh_exo["CDD"]= np.nan
        hh_exo["intercept"] =1
        hh_exo_final = hh_exo.copy(deep = True)
        hh_elec.columns = ["Total_daily_elec"]
        years_min_temp = int(hh_exo["temp_avg"].min())
        years_max_temp = int(hh_exo["temp_avg"].max())
        cooling_points = np.arange(60,years_max_temp+1)
        heating_points = np.arange(years_min_temp,66)
        
        #Loop uses the balance point range to find the best balance point 
        for HBP_idx,HBP in enumerate(heating_points):
            #Criteria to ensure the heating balance point (HBP) is appropriate
            hh_exo.loc[:,"HDD"] = (HBP- hh_exo["temp_avg"]).clip(lower=0) #pd.clip trims the value at the specified threshold to 
            hh_HDD = hh_exo["HDD"].copy(deep= True)
            
            #Makes sure there are 10 observations and 20 degree days
            if (hh_HDD[hh_HDD>0].count()>10 and hh_HDD[hh_HDD>0].sum()>20): 
                for CBP_idx,CBP in enumerate(cooling_points):                 
                    #Criteria to ensure the cooling balance point is appropriate
                    hh_exo.loc[:,"CDD"] = (hh_exo["temp_avg"]-CBP).clip(lower=0) 
                    hh_CDD = hh_exo["CDD"].copy(deep= True)
                    
                    #Makes sure there are 10 observations and 20 degree days
                    if (hh_CDD[hh_CDD>0].count()>10 and hh_HDD[hh_HDD>0].sum()>20): 

                        #Use OLS to explore the models 
                        #fits OLS regression using temperatures at or above the cooling point range
                        hh_exo.iloc[:,2:-5] = hh_exo.iloc[:,2:-5].astype(np.uint8)
                        
                            
                        mod0_exo =  hh_exo.drop(axis=1,labels =["avg_rel_humidity",
                                     "temp_avg","BILACCT_K","elec_cost"])   
                        mod0_results = sm.OLS(hh_elec.astype(float),
                                              mod0_exo.astype(float)).fit()
                        day_drop = mod0_results.pvalues[1:8][mod0_results.pvalues[1:8]==mod0_results.pvalues[1:8].max()].index[0]
                        mon_drop = mod0_results.pvalues[8:19][mod0_results.pvalues[8:19]==mod0_results.pvalues[8:19].max()].index[0]
                        if (mod0_results.pvalues["CDD"]<0.1 and mod0_results.pvalues["HDD"]<0.1 and\
                            mod0_results.pvalues["intercept"]<0.1 and mod0_results.params["CDD"]>0 \
                                and mod0_results.params["HDD"]>0 and mod0_results.params["intercept"]>0):#Check that the points are statistically significiant (10%)
                            mod1_exo =  hh_exo.drop(axis=1,labels =["avg_rel_humidity",
                                         "temp_avg","BILACCT_K","elec_cost"]).drop(axis=1,labels=[day_drop,mon_drop])   
                            mod1_results = sm.OLS(hh_elec.astype(float),
                                                  mod1_exo.astype(float)).fit()
                            mod2_exo =  hh_exo.drop(axis=1,
                                                    labels =["temp_avg",
                                                             "BILACCT_K",
                                                             "elec_cost"]).drop(
                                            axis=1,labels=[day_drop,mon_drop])     
                            mod2_results = sm.OLS(hh_elec.astype(float),
                                                  mod2_exo.astype(float)).fit()
                            
                            mod3_exo =  hh_exo.drop(axis=1,labels =[
                                         "temp_avg","BILACCT_K"]).drop(axis=1,
                                                                       labels=[day_drop,mon_drop])
                                                                       
                            mod3_results = sm.OLS(hh_elec.astype(float),
                                                  mod3_exo.astype(float)).fit()
                            mod4_exo =  hh_exo.drop(axis=1,labels =[
                                         "temp_avg","BILACCT_K","avg_rel_humidity"]).drop(axis=1,labels=[day_drop,mon_drop])   
                            mod4_results = sm.OLS(hh_elec.astype(float),
                                                  mod4_exo.astype(float)).fit()
                            
                        
                            mod_entries.append([HBP,CBP,
                                                mod1_results.rsquared_adj,
                                                mod2_results.rsquared_adj,
                                                mod3_results.rsquared_adj,
                                                mod4_results.rsquared_adj,
                                                day_drop,mon_drop])
                        else: #if the balance points' pvalues are greater than 10% then move to the next step without saving
                            continue

                #If the hh_CDD criteria is not met then continue to the next CBP
                else:
                    continue
            #If the hh_HDD criteria is not met then continue to the next HBP
            else:
                continue
        
        mod_entries_eval = pd.DataFrame(data=mod_entries,columns = ["HBP","CBP","Mod1","Mod2","Mod3","Mod4","day","month"])
        #Determines which model is the best (max requared adjusted)
        if mod_entries_eval.empty:
            continue #skips the model evaluation and CV if the dataframe does not have any information
        else:
            best_parameters = mod_entries_eval[["HBP","CBP","day","month"]][(mod_entries_eval.Mod1==mod_entries_eval.iloc[:,2:5].max().max()) |
                             (mod_entries_eval.Mod2==mod_entries_eval.iloc[:,2:5].max().max())|
                             (mod_entries_eval.Mod3==mod_entries_eval.iloc[:,2:5].max().max())|
                             (mod_entries_eval.Mod4==mod_entries_eval.iloc[:,2:5].max().max())].iloc[0]
            
            best_mod = mod_entries_eval.loc[:,mod_entries_eval[
                mod_entries_eval.iloc[:,2:5].eq(mod_entries_eval.iloc[:,2:5].max().max())].any()].columns[0]
            
            best_model_parameters.loc[households,["HBP","CBP","day","month"]]= best_parameters.values
            best_model_parameters.loc[households,"best_model"]= best_mod
            all_models_adjr2.loc[households,:] = mod_entries_eval[["Mod1","Mod2","Mod3", "Mod4"]][(mod_entries_eval.Mod1==mod_entries_eval.iloc[:,2:5].max().max()) |
                             (mod_entries_eval.Mod2==mod_entries_eval.iloc[:,2:5].max().max())|
                             (mod_entries_eval.Mod3==mod_entries_eval.iloc[:,2:5].max().max())|
                             (mod_entries_eval.Mod4==mod_entries_eval.iloc[:,2:5].max().max())].iloc[0].values

#-----------------------Cross Validation testing-------------------------------
            hh_exo_cv = piece_exogen[piece_exogen["BILACCT_K"]==int(households)].set_index(
                piece_exogen[piece_exogen["BILACCT_K"]==int(households)].index).dropna(axis=0)
     
            hh_exo_cv = hh_exo_cv.groupby(hh_exo_cv.index.date).mean()
            hh_exo_cv["HDD"]= np.nan
            hh_exo_cv["CDD"]= np.nan
            hh_exo_cv["intercept"] =1
            
            hh_exo_cv.loc[:,"HDD"] = (best_parameters["HBP"]- hh_exo_cv["temp_avg"]).clip(lower=0) 
            hh_exo_cv.loc[:,"CDD"] = (hh_exo_cv["temp_avg"]-best_parameters["CBP"]).clip(lower=0) 
            hh_exo_cv.iloc[:,2:-5] = hh_exo_cv.iloc[:,2:-5].astype(np.uint8)
            
            mod1_exo_cv =  hh_exo_cv.drop(axis=1,labels =["avg_rel_humidity",
                          "temp_avg","BILACCT_K","elec_cost"]).drop(axis=1,labels=[best_parameters["day"],
                                                                                            best_parameters["month"]])
            
            mod2_exo_cv =  hh_exo_cv.drop(axis=1,labels =[
                          "temp_avg","BILACCT_K","elec_cost"]).drop(axis=1,labels=[best_parameters["day"],
                                                                                            best_parameters["month"]])     
           
            
            mod3_exo_cv =  hh_exo_cv.drop(axis=1,labels =[
                          "temp_avg","BILACCT_K"]).drop(axis=1,labels=[best_parameters["day"],
                                                                                            best_parameters["month"]])
                                                                       
            mod4_exo_cv =  hh_exo_cv.drop(axis=1,labels =[
                          "temp_avg","BILACCT_K","avg_rel_humidity"]).drop(axis=1,labels=[best_parameters["day"],
                                                                                            best_parameters["month"]])   

            nfolds = folds
            cv = RepeatedKFold(n_splits = nfolds, n_repeats=n_repeats,random_state = 1) 
                
            #Model 1: 5 Point Regression
            model = LinearRegression()
            
            # cross-validation scores (5-point regression with fixed effects)
            rMSE.loc[hh_elec.name,"mod1"] = np.abs(np.mean(cross_val_score(model,
                                                        mod1_exo_cv, 
                                      hh_elec,
                                      scoring ='neg_root_mean_squared_error', 
                                      cv=cv, n_jobs=-1)))
            # cross-validation scores (Quadratic regression)
            rMSE.loc[hh_elec.name,"mod2"] =  np.abs(np.mean(cross_val_score(model,
                                                      mod2_exo_cv,
                                                      hh_elec,
                                      scoring ='neg_root_mean_squared_error', 
                                      cv=cv, n_jobs=-1)))
            
            # cross-validation scores (Quadratic regression)
            rMSE.loc[hh_elec.name,"mod3"] =  np.abs(np.mean(cross_val_score(model,
                                                      mod3_exo_cv,
                                                      hh_elec,
                                      scoring ='neg_root_mean_squared_error', 
                                      cv=cv, n_jobs=-1)))
            rMSE.loc[hh_elec.name,"mod4"] =  np.abs(np.mean(cross_val_score(model,
                                                      mod4_exo_cv,
                                                      hh_elec,
                                      scoring ='neg_root_mean_squared_error', 
                                      cv=cv, n_jobs=-1)))
                        
            
        # Repeating the process to ensure I am capturing the right variables
        hh_exo_final.loc[:,"HDD"] = (best_model_parameters.loc[households,"HBP"]- hh_exo["temp_avg"]).clip(lower=0) #pd.clip trims the value at the specified threshold to 
        hh_exo_final.loc[:,"CDD"] = (hh_exo["temp_avg"]-best_model_parameters.loc[households,"CBP"]).clip(lower=0) 
        mod_exo_final =  hh_exo_final.drop(axis=1,labels =["avg_rel_humidity",
                                     "temp_avg","BILACCT_K"]).drop(axis=1,labels=[
                                         best_model_parameters.loc[households,"day"],
                                         best_model_parameters.loc[households,"month"]])   
        mod_final_results = sm.OLS(hh_elec.astype(float),
                              mod_exo_final.astype(float)).fit()
        mod_final_results.summary()
        predictions.loc[hh_elec.index,households] = mod_final_results.predict()
        
        five_pt_models.loc[households,c_year] = mod_final_results
        five_pt_models1.loc[households,c_year] = mod1_results
        five_pt_models2.loc[households,c_year] = mod2_results
        five_pt_models3.loc[households,c_year] = mod3_results
        five_pt_models4.loc[households,c_year] = mod4_results

        piecewise_model_r_square_adj.loc[households,:] = mod_final_results.rsquared_adj
        piecewise_model_coefficient.loc[households,mod_final_results.params.index] = mod_final_results.params.values
        piecewise_model_p_values.loc[households,mod_final_results.pvalues.index] = mod_final_results.pvalues.values
    
    #Model 4 is the same as the best model that includes the electricity price 
    return(piecewise_model_coefficient,#0
           piecewise_model_r_square_adj,#1
           five_pt_models,#2
            predictions,#3,
            all_models_adjr2, #4
            best_model_parameters,#5
            rMSE,#6
            piecewise_model_p_values,#7,
            five_pt_models1,#8
            five_pt_models2,#9
            five_pt_models3,#10
            
            )

year_labels =  ["15/16", "16/17", "17/18", "18/19"]

#Saves the models
[five_pt_1516_coef,
five_pt_1516_adj_r2,
five_pt_1516_model,
five_pt_1516_pre,
five_pt_1516_all_mods_adj_r2,
five_pt_1516_best_model_parameters,
five_pt_1516_rmse,
five_pt_1516_pvalues,
five_pt_1516_model_basic,
five_pt_1516_model_humidity,
five_pt_1516_model_hud_ele] = five_parameter_regression("05-01-2015", "04-30-2016",10,10, year_labels[0])

[five_pt_1617_coef,
 five_pt_1617_adj_r2,
 five_pt_1617_model,
five_pt_1617_pre,
five_pt_1617_all_mods_adj_r2,
five_pt_1617_best_model_parameters,
five_pt_1617_rmse,
five_pt_1617_pvalues,
five_pt_1617_model_basic,
five_pt_1617_model_humidity,
five_pt_1617_model_hud_el]  = five_parameter_regression("05-01-2016", "04-30-2017",10,10,year_labels[1])

[five_pt_1718_coef,
five_pt_1718_adj_r2,
five_pt_1718_model,
five_pt_1718_pre,
five_pt_1718_all_mods_adj_r2,
five_pt_1718_best_model_parameters,
five_pt_1718_rmse,
five_pt_1718_pvalues,
five_pt_1718_model_basic,
five_pt_1718_model_humidity,
five_pt_1718_model_hud_el]  = five_parameter_regression("05-01-2017", "04-30-2018",10,10,year_labels[2])

[five_pt_1819_coef,
 five_pt_1819_adj_r2,
 five_pt_1819_model,
five_pt_1819_pre,
five_pt_1819_all_mods_adj_r2,
five_pt_1819_best_model_parameters,
five_pt_1819_rmse,
five_pt_1819_pvalues,
five_pt_1819_model_basic,
five_pt_1819_model_humidity,
five_pt_1819_model_hud_el] = five_parameter_regression("05-01-2018", "04-30-2019",10,10, year_labels[3])
    
def VDD_individuals():
    best_para_frame_CBP = pd.concat([
                       five_pt_1516_best_model_parameters.loc[:,"CBP"],
                       five_pt_1617_best_model_parameters.loc[:,"CBP"],
                       five_pt_1718_best_model_parameters.loc[:,"CBP"],
                       five_pt_1819_best_model_parameters.loc[:,"CBP"]],axis=1)
    best_para_frame_CBP.columns = ["15/16","16/17","17/18","18/19"]

    best_para_frame_HBP = pd.concat([
                       five_pt_1516_best_model_parameters.loc[:,"HBP"],
                       five_pt_1617_best_model_parameters.loc[:,"HBP"],
                       five_pt_1718_best_model_parameters.loc[:,"HBP"],
                       five_pt_1819_best_model_parameters.loc[:,"HBP"]],axis=1)
    best_para_frame_HBP.columns = ["15/16","16/17","17/18","18/19"]

    years_range_mods= pd.DataFrame(data=[
        pd.date_range(start= "05-01-2015",  periods = 4,
                      freq=pd.DateOffset(years=1)),
        pd.date_range(start= "04-30-2016",  periods = 4,
                      freq=pd.DateOffset(years=1))],
        ).T
    years_range_mods.columns = ["start_date", "end_date"]
    temps_daily = exogen["temp_avg"].groupby(exogen.index.date).mean()
    
    """Function to calculate variable-base degree days for each household"""
    CDD_df = pd.DataFrame(index=best_para_frame_CBP.index,
                          columns=best_para_frame_CBP.columns)
    HDD_df = pd.DataFrame(index=best_para_frame_CBP.index,
                          columns=best_para_frame_CBP.columns)
    share_of_days_cooling = pd.DataFrame(index=best_para_frame_CBP.index,
                          columns=best_para_frame_CBP.columns)
    for year_row, year_lab in enumerate(CDD_df.columns):
        date_range = years_range_mods.loc[year_row,:]
        temps_year_daily = temps_daily.loc[date_range[0]:date_range[1]]
        for row,house in enumerate(best_para_frame_CBP.loc[best_para_frame_CBP.index,
                                                           year_lab].dropna().index):
            CBP_point = best_para_frame_CBP.loc[house,year_lab]
            HBP_point = best_para_frame_HBP.loc[house,year_lab]
            CDD_df.loc[house,year_lab]= temps_year_daily.subtract(
                CBP_point).clip(lower=0).sum()
            HDD_df.loc[house,year_lab]= (HBP_point-temps_year_daily).clip(lower=0).sum()
            temps_year_daily
            share_of_days_cooling.loc[house,year_lab]= temps_year_daily[
                temps_year_daily>CBP_point].count()/temps_year_daily.count()
            
    return(CDD_df,HDD_df,share_of_days_cooling)
[household_CDD,household_HDD,
 share_of_days_cooling] = VDD_individuals()    

def store_baseline_data():
    annual_cooling_points = pd.concat([
                                       five_pt_1516_best_model_parameters["CBP"],
                                       five_pt_1617_best_model_parameters["CBP"],
                                        five_pt_1718_best_model_parameters["CBP"],
                                        five_pt_1819_best_model_parameters["CBP"]
                                       ]
                                      , axis=1).astype(float)
    
    annual_cooling_points.columns = ["15/16", "16/17", "17/18", "18/19"]
    
    #Heating balance points
    annual_heating_points = pd.concat([ five_pt_1516_best_model_parameters["HBP"],
                                       five_pt_1617_best_model_parameters["HBP"],
                                        five_pt_1718_best_model_parameters["HBP"],
                                        five_pt_1819_best_model_parameters["HBP"]
                                       ]
                                      , axis=1).astype(float)
    
    annual_heating_points.columns = ["15/16", "16/17", "17/18", "18/19"]
    
    models_2015_2019 = pd.concat([
                                  five_pt_1516_model["15/16"],
                                  five_pt_1617_model["16/17"],
                                  five_pt_1718_model["17/18"],
                                  five_pt_1819_model["18/19"]],
                                  axis=1)
    models_2015_2019_basic = pd.concat([
                                  five_pt_1516_model_basic["15/16"],
                                  five_pt_1617_model_basic["16/17"],
                                  five_pt_1718_model_basic["17/18"],
                                  five_pt_1819_model_basic["18/19"]],
                                  axis=1)
    models_2015_2019_humidity = pd.concat([
                                  five_pt_1516_model_humidity["15/16"],
                                  five_pt_1617_model_humidity["16/17"],
                                  five_pt_1718_model_humidity["17/18"],
                                  five_pt_1819_model_humidity["18/19"]],
                                  axis=1)
    # models_2015_2019_hud_el = pd.concat([
    #                               five_pt_1516_model_hud_el["15/16"],
    #                               five_pt_1617_model_hud_el["16/17"],
    #                               five_pt_1718_model_hud_el["17/18"],
    #                               five_pt_1819_model_hud_el["18/19"]],
    #                               axis=1)
    #---------------------------Temperature Response Coefficients-----------------
    #Cooling Slopes
    CDD_coeff = pd.concat([
                           five_pt_1516_coef["CDD"],
                           five_pt_1617_coef["CDD"],
                           five_pt_1718_coef["CDD"],
                           five_pt_1819_coef["CDD"]
                               ],axis=1)
    CDD_coeff.columns = year_labels
    
    #--------------------------Heating Slopes--------------------------------------
    HDD_coeff = pd.concat([
                           five_pt_1516_coef["HDD"],
                           five_pt_1617_coef["HDD"],
                           five_pt_1718_coef["HDD"],
                           five_pt_1819_coef["HDD"]
                               ],axis=1)
    HDD_coeff.columns = year_labels
    
    #----------------------Intercepts----------------------------------------------
    intercept_coef = pd.concat([
                           five_pt_1516_coef["intercept"],
                           five_pt_1617_coef["intercept"],
                           five_pt_1718_coef["intercept"],
                           five_pt_1819_coef["intercept"]
                               ],axis=1)
    
    intercept_coef.columns = year_labels
    
    best_para_frame_CBP = pd.concat([
                       five_pt_1516_best_model_parameters.loc[:,"CBP"],
                       five_pt_1617_best_model_parameters.loc[:,"CBP"],
                       five_pt_1718_best_model_parameters.loc[:,"CBP"],
                       five_pt_1819_best_model_parameters.loc[:,"CBP"]],axis=1)
    best_para_frame_CBP.columns = ["15/16","16/17","17/18","18/19"]
    
    best_para_frame_HBP = pd.concat([
                       five_pt_1516_best_model_parameters.loc[:,"HBP"],
                       five_pt_1617_best_model_parameters.loc[:,"HBP"],
                       five_pt_1718_best_model_parameters.loc[:,"HBP"],
                       five_pt_1819_best_model_parameters.loc[:,"HBP"]],axis=1)
    best_para_frame_HBP.columns = ["15/16","16/17","17/18","18/19"]
    
    years_range_mods= pd.DataFrame(data=[
        pd.date_range(start= "05-01-2015",  periods = 4,
                      freq=pd.DateOffset(years=1)),
        pd.date_range(start= "04-30-2016",  periods = 4,
                      freq=pd.DateOffset(years=1))],
        ).T
    years_range_mods.columns = ["start_date", "end_date"]
    
    
    five_pt_1516_best_model_parameters.to_excel(path+r'/Data/Baseline/2015_2016/2015_2016_best_paramters.xlsx')
    five_pt_1617_best_model_parameters.to_excel(path+r'/Data/Baseline/2016_2017/2016_2017_best_paramters.xlsx')
    five_pt_1718_best_model_parameters.to_excel(path+r'/Data/Baseline/2017_2018/2017_2018_best_paramters.xlsx')
    five_pt_1819_best_model_parameters.to_excel(path+r'/Data/Baseline/2018_2019/2018_2019_best_paramters.xlsx')

    five_pt_1516_coef.to_excel(path+r'/Data/Baseline/2015_2016/2015_2016_coefficients.xlsx')
    five_pt_1617_coef.to_excel(path+r'/Data/Baseline/2016_2017/2016_2017_coefficients.xlsx')
    five_pt_1718_coef.to_excel(path+r'/Data/Baseline/2017_2018/2017_2018_coefficients.xlsx')
    five_pt_1819_coef.to_excel(path+r'/Data/Baseline/2018_2019/2018_2019_coefficients.xlsx')
    
    five_pt_1516_adj_r2.to_excel(path+'/Data/Baseline/2015_2016/2015_2016_adjr2.xlsx')
    five_pt_1617_adj_r2.to_excel(path+'/Data/Baseline/2016_2017/2016_2017_adjr2.xlsx')
    five_pt_1718_adj_r2.to_excel(path+'/Data/Baseline/2017_2018/2017_2018_adjr2.xlsx')
    five_pt_1819_adj_r2.to_excel(path+'/Data/Baseline/2018_2019/2018_2019_adjr2.xlsx')
    
    five_pt_1516_all_mods_adj_r2.to_excel(path+'/Data/Baseline/2015_2016/2015_2016_adjr2.xlsx')
    five_pt_1617_all_mods_adj_r2.to_excel(path+'/Data/Baseline/2016_2017/2016_2017_adjr2.xlsx')
    five_pt_1718_all_mods_adj_r2.to_excel(path+'/Data/Baseline/2017_2018/2017_2018_adjr2.xlsx')
    five_pt_1819_all_mods_adj_r2.to_excel(path+'/Data/Baseline/2018_2019/2018_2019_adjr2.xlsx')
    
    
    five_pt_1516_rmse.to_excel(path+'/Data/Baseline/2015_2016/2015_2016_rmse.xlsx')
    five_pt_1617_rmse.to_excel(path+'/Data/Baseline/2016_2017/2016_2017_rmse.xlsx')
    five_pt_1718_rmse.to_excel(path+'/Data/Baseline/2017_2018/2017_2018_rmse.xlsx')
    five_pt_1819_rmse.to_excel(path+'/Data/Baseline/2018_2019/2018_2019_rmse.xlsx')
    
    #Model dataframes 
    
    models_2015_2019.to_pickle(path+'/Data/Baseline/2015_2019_regression_models.pkl')
    models_2015_2019_basic.to_pickle(path+'/Data/Baseline/2015_2019_regression_models_basic.pkl')
    models_2015_2019_humidity.to_pickle(path+'/Data/Baseline/2015_2019_regression_models_humidity.pkl')
    
    models_2015_2019.to_pickle(path+'/Data/Baseline/2015_2019_regression_models.pkl')
    
    share_of_days_cooling.to_excel(path+'/Data/Baseline/household_share.xlsx')
    household_CDD.to_excel(path+'/Data/Baseline//household_VBCDD.xlsx')
    household_HDD.to_excel(path+'/Data/Baseline/household_VBHDD.xlsx')
    
    annual_cooling_points.to_csv(path+'/Data/Baseline/Individual_Cooling_Balance_points.csv')
    annual_heating_points.to_csv(path+'/Data/Baseline/Individual_Heating_Balance_points.csv')
    
    intercept_coef.to_excel(path+'/Data/Baseline/2015_2019_intercept_coefficients.xlsx')
    HDD_coeff.to_excel(path+'/Data/Baseline/2015_2019_HDD_coefficients.xlsx' )
    CDD_coeff.to_excel(path+'/Data/Baseline/2015_2019_CDD_coefficients.xlsx')

store_baseline_data()

electric_households = survey_all["BILACCT_K"][~(
    (survey_all.VHEATEQUIP =="Separate gas furnace") |
    (survey_all.VHEATEQUIP =='Gas furnace packaged with AC unit (sometimes called a gas pa)')|
    (survey_all.VACTYPE =='AC unit packaged with gas heating (sometimes called a gas pa')
                                                   )].values


def filtered_hh(data1):
    filtered_data = data1.loc[
        data1.index.isin(electric_households),:]
    return(filtered_data)

def main():
    rmse_df = pd.concat([
        five_pt_1516_rmse["mod4"].dropna(),five_pt_1617_rmse["mod4"].dropna(),
        five_pt_1718_rmse["mod4"].dropna(),five_pt_1819_rmse["mod4"].dropna()],
        axis=1).astype(float)
    rmse_df = filtered_hh(data1 =rmse_df)

    adjr2_df = pd.concat([
        five_pt_1516_adj_r2.dropna(),five_pt_1617_adj_r2.dropna(),
        five_pt_1718_adj_r2.dropna(),five_pt_1819_adj_r2.dropna()],
        axis=1).astype(float)
    adjr2_df = filtered_hh(data1 =adjr2_df)

    rmse_df.columns= ["2015-2016", "2016-2017","2017-2018","2018-2019"]
    adjr2_df.columns= ["2015-2016", "2016-2017","2017-2018","2018-2019"]
    
    rmse_df.reset_index(inplace= True)
    adjr2_df.reset_index(inplace= True)
    Table1 = pd.concat([rmse_df.iloc[:,1:].agg({"count","mean","min","max"}).T,
                        adjr2_df.iloc[:,1:].multiply(100).agg({"mean","min","max"}).T],
                       axis=1)
    Table1.to_excel(path+'/Results/Table1.xlsx')
    return(Table1)
if __name__=="__main__":
    main()
