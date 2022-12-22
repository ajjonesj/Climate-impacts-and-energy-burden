# -*- coding: utf-8 -*-
"""
This file calls the daily electricity data and survey data and makes the necessary changes and format
"""
import pandas as pd
import numpy as np
 
path = 'C:/Users/andre/Box/Andrew Jones_PhD Research/Climate change impacts on future electricity consumption and energy burden'

survey_all = pd.read_stata('C:/Users/andre/Box/RAPID COVID Residential Energy/Arizona_Full datasets/Arizona 2019-2020 dataset/RET_2017_part1.dta')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7,4,5,6,2,1,8,3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"]!=""].unique()):
    survey_all.loc[survey_all["VINCOME"]==ig_num,["VINCOME_new"]] = int(ig_unsorted[idxs])
def weather_data():
    humidity_data = pd.read_csv(path+'Data/humdity_2014_2019_data.csv',
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
#%%Data Processing 
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



def main():
    exogen.to_csv(path+'/Data/exogen_data_mod.csv')
    panel_data.to_csv(path+ '/Data/Daily_consumption.csv')
    
if __name__=="__main__":
    main()