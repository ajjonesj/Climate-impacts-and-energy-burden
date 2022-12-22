# -*- coding: utf-8 -*-
"""
Write the discription here
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from functions_and_labels import add_race_income
from functions_and_labels import labels,filtered_hh 

path = 'C:/Users/andre/Box/Andrew Jones_PhD Research/Climate change impacts on future electricity consumption and energy burden'

#%%Data Processing 
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
panel_data = pd.read_csv(path+'/Data/Daily_consumption.csv', parse_dates=[0],
                       index_col=[0])

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
#Removes households without an AC SEER rating
survey_AC_efficiency = survey.loc[survey["VACSER1"].dropna().index,:]

#Run this line of code to remove the households that left their income blank
survey_AC_efficiency = survey_AC_efficiency[(survey_AC_efficiency.VINCOME !="")]
survey_AC_efficiency["VHEATEQUIP"].unique()
survey_AC_efficiency.groupby("VINCOME").count()

share_of_days_cooling = pd.read_excel(path + '/Data/Baseline/household_share.xlsx').set_index("BILACCT_K")

X = survey_all[survey_all.BILACCT_K.isin(filtered_hh(CDD_coeff).index)]

X.set_index("BILACCT_K", inplace=True)
X.loc[:,"Share"] = filtered_hh(share_of_days_cooling)["17/18"]

X_models = X.loc[:,["VACSER1",'VFANNUM','VHOUSEHOLD','VINCOME','VINCOME_new','VACTYPE',
                    'VACUNITS','VTHERMOSTAT', "Share",
                    'VBANDWELL', 'VBANSQFEET',"VRESAGE"]]
percent_of_houses_SEER_oob = (X_models["VACSER1"][(X_models["VACSER1"]>28) | (X_models["VACSER1"]<7)].count()/\
    X_models["VACSER1"].dropna().count())* 100
X_models["Intercept"] = 1   
# X_models.set_index("BILACCT_K", inplace= True)
y = CDD_coeff.loc[:,"17/18"]
marker_styles = ['+','s', '.', 'v','*', 'h','x','d' ]
[income_group_axis_labels,income_group_numbers,income_group_labels,
 income_group_legend_labels,year_for_graphs] = labels()

#%%------------------------------------------------------------------------------
#20/60/20 rule
#------------------------------------------------------------------------------
def data_processing_rule(X_models):
    """function used to do the 20/60/20 rule"""
    SEER_data = X_models[~(X_models.VACSER1.isna())]
    cases = pd.DataFrame(data = SEER_data.index)
    order = pd.DataFrame(np.tile(np.arange(1,6),
                                 [int(np.ceil(len(cases)/5)),1]))
   # https://stackoverflow.com/questions/65944937/how-can-i-repeat-an-array-m-times
    if len(order) == len(order.stack().values):
        cases["order"] = order.stack().values
    else:
        len(order.stack().values) -len(cases)
        s = pd.Series(order.stack().values)
        cases["order"] = s[:-4]
    #------------------------------------------------------------------
    first20 = cases.loc[cases.order ==1,"BILACCT_K"]
    middle60 = cases.loc[(cases.order >=2) &(cases.order <=4),"BILACCT_K"]
    last20 = cases.loc[cases.order ==5,"BILACCT_K"]
    first80 = cases.loc[cases.order <5,"BILACCT_K"]
    last80 = cases.loc[(cases.order >=2) &(cases.order <=5),"BILACCT_K"]
    return(first20,#0
           middle60,#1
           last20,#2
           first80,#3
           last80#4
           )

def drop_empty_OLS_variables(mod2_y, mod2_X):
    accts_mod2 = mod2_y[mod2_y.index.isin(
                       mod2_X.dropna().index.values)].dropna().index.values
    mod2_y = mod2_y.loc[accts_mod2]
    mod2_X = mod2_X.loc[accts_mod2,:]
    return(mod2_y,mod2_X)


#%% Training and testing
first80_accounts = data_processing_rule(X_models)[3]
X_first80 = X_models.loc[first80_accounts,:]
y_first80 = y.loc[first80_accounts]

#quick test to see if I am thinking about this problem better
training_X = X_first80[["Intercept","Share","VFANNUM","VACSER1"]]

training_X = pd.concat([training_X,
                    #Cooling Infrastructure
                    pd.get_dummies(X_first80["VACUNITS"][X_first80["VACUNITS"]!= ""],
                                   prefix= "AC_units"),
                    pd.get_dummies(X_first80["VACTYPE"][(
                        (X_first80["VACTYPE"]!="")& 
                        (X_first80["VACTYPE"]!="Central-Gas"))]),
                    #Housing Infrastructure  
                    X_first80["VHOUSEHOLD"],
                    X_first80["VRESAGE"],
                    pd.get_dummies(X_first80["VBANSQFEET"][
                          X_first80["VBANSQFEET"] !=""], prefix= "Sqft"),
                     pd.get_dummies(X_first80["VBANDWELL"][
                          X_first80["VBANDWELL"] != ""])
                    ],axis=1)

training_y = y_first80
[training_y,training_X ] = drop_empty_OLS_variables(training_y, training_X)

training_X.drop(axis=1, labels=["AC_units_One",
                                "Central-Heat pump",
                            "Single family home",
                            "Sqft_Less than 1,500"], inplace= True)


training_model_robust = sm.OLS(training_y.reset_index(drop=True),
                        training_X.reset_index(drop= True)).fit(cov_type='HC0')

training_model_robust.save(path + '/CDD_slopes_models.pkl')

from statsmodels.iolib.summary2 import summary_col

tab3 = summary_col(training_model_robust,
                       info_dict= {"N":lambda x:(x.nobs)},
                       stars=True,float_format="%.3f", 
                       model_names=(["Model 4"]),
                       regressor_order=[
                           # SEER
                           'VACSER1',                         
                           #Temperature Indirect Effect (Share of Days)
                           'Share',                          
                           #Cooling Infrastructure
                           'AC_units_3 or more',
                           'AC_units_Two','Central-Separate AC',
                           'Central-Unknown','VFANNUM',
                           #Housing Infrastructure
                           'Apartment/Condo/Townhouse','Mobile home',
                           'Sqft_1,500 - 2,999','Sqft_3,000 or more',
                           'VHOUSEHOLD','VRESAGE',
                           #Income
                           '$15,000 to $24,999',
                           '$25,000 to $34,999','$35,000 to $49,999','$50,000 to $74,999',
                           '$75,000 to $99,999','$100,000 to $149,999',
                           '$150,000 or more',
                           #Other Important Factor s
                           'Intercept','N','R-squared',
                           'R-squared Adj.'] )
tab3.tables[0].to_excel(path+"/Results/Table3.xlsx")
print(tab3)

#%%Summary Table 
def Cooling_slopes_summary_table():
    #Model 1: SEER 
    training_XM1 = X_first80[["Intercept","VACSER1"]]
    training_yM1 = y_first80
    [training_yM1,training_XM1] = drop_empty_OLS_variables(training_yM1 , training_XM1)
    
    training_modelM1 = sm.OLS(training_yM1.reset_index(drop=True),
                            training_XM1.reset_index(drop= True)).fit(cov_type='HC0')
    #------------------------------------------------------------------------------
    #Model 2: Share and SEER
    training_XM2 = X_first80[["Intercept", "Share","VACSER1"]]
    training_yM2 = y_first80
    [training_yM2,training_XM2 ] = drop_empty_OLS_variables(training_yM2 , training_XM2)
    training_modelM2 = sm.OLS(training_yM2.reset_index(drop=True),
                            training_XM2.reset_index(drop= True)).fit(cov_type='HC0')
    #----------------------------------------------------------------------------
    #Model 3: Share and SEER AND Cooling Infrastructure
    training_XM3 = X_first80[["Intercept", "Share","VACSER1","VFANNUM"]]
    training_yM3 = y_first80
    
    training_XM3 = pd.concat([training_XM3,
                        #Cooling Infrastructure
                        pd.get_dummies(X_first80["VACUNITS"][X_first80["VACUNITS"]!= ""],
                                       prefix= "AC_units"),
                        pd.get_dummies(X_first80["VACTYPE"][(
                            (X_first80["VACTYPE"]!="")& 
                            (X_first80["VACTYPE"]!="Central-Gas"))]),
                        ],axis=1)
    [training_yM3,training_XM3 ] = drop_empty_OLS_variables(training_yM3 , training_XM3)
    training_XM3.drop(axis=1, labels=["AC_units_One",
                                    "Central-Heat pump"], inplace= True)
    training_modelM3 = sm.OLS(training_yM3.reset_index(drop=True),
                            training_XM3.reset_index(drop= True)).fit(cov_type='HC0')
    training_modelM3.summary()
    #------------------------------------------------------------------------------
    #Model 4: Share and SEER AND Cooling  and Housing Infrastructure
    training_XM4 = X_first80[["Intercept", "Share","VFANNUM","VACSER1"]]
    training_yM4 = y_first80
    
    training_XM4 = pd.concat([training_XM4,
                        #Cooling Infrastructure
                        pd.get_dummies(X_first80["VACUNITS"][X_first80["VACUNITS"]!= ""],
                                       prefix= "AC_units"),
                        pd.get_dummies(X_first80["VACTYPE"][(
                            (X_first80["VACTYPE"]!="")& 
                            (X_first80["VACTYPE"]!="Central-Gas"))]),
                        #Housing Infrastructure  
                        X_first80["VHOUSEHOLD"],
                        X_first80["VRESAGE"],
                        pd.get_dummies(X_first80["VBANSQFEET"][
                              X_first80["VBANSQFEET"] !=""], prefix= "Sqft"),
                         pd.get_dummies(X_first80["VBANDWELL"][
                              X_first80["VBANDWELL"] != ""])
                        ],axis=1)
    [training_yM4,training_XM4 ] = drop_empty_OLS_variables(training_yM4 , training_XM4)
    training_XM4.drop(axis=1, labels=["AC_units_One",
                                    "Central-Heat pump",
                                    "Single family home",
                                    "Sqft_Less than 1,500"], inplace= True)
    training_modelM4 = sm.OLS(training_yM4.reset_index(drop=True),
                            training_XM4.reset_index(drop= True)).fit(cov_type='HC0')
    #------------------------------------------------------------------------------
    #Model 5: Share and SEER AND Cooling  and Housing Infrastructure and Income 
    
    
    training_XM5 = X_first80[["Intercept", "Share","VFANNUM","VACSER1"]]
    
    training_XM5 = pd.concat([training_XM5,
                        #Cooling Infrastructure
                        pd.get_dummies(X_first80["VINCOME"][X_first80["VINCOME"]!=""]),
                        pd.get_dummies(X_first80["VACUNITS"][X_first80["VACUNITS"]!= ""],
                                       prefix= "AC_units"),
                        pd.get_dummies(X_first80["VACTYPE"][(
                            (X_first80["VACTYPE"]!="")& 
                            (X_first80["VACTYPE"]!="Central-Gas"))]),
                        #Housing Infrastructure  
                        X_first80["VHOUSEHOLD"],
                        X_first80["VRESAGE"],
                        pd.get_dummies(X_first80["VBANSQFEET"][
                              X_first80["VBANSQFEET"] !=""], prefix= "Sqft"),
                         pd.get_dummies(X_first80["VBANDWELL"][
                              X_first80["VBANDWELL"] != ""])],axis=1)
    
    training_yM5 = y_first80
    [training_yM5,training_XM5 ] = drop_empty_OLS_variables(training_yM5 , training_XM5)
    training_XM5.drop(axis=1, labels=["Less than $15,000","AC_units_One",
                                    "Central-Heat pump",
                                "Single family home",
                                "Sqft_Less than 1,500"], inplace= True)
    training_modelM5 = sm.OLS(training_yM5.reset_index(drop=True),
                            training_XM5.reset_index(drop= True)).fit()
    
    dfoutput = summary_col([training_modelM1, training_modelM2, training_modelM3,
                            training_modelM4, training_modelM5 ],
                           info_dict= {"N":lambda x:(x.nobs),
                                       'Cov. Type:': lambda x: x.fit_options['HC0']},
                           stars=True,float_format="%.3f", 
                           model_names=(["Model 1", "Model 2","Model 3", "Model 4", "Model 5","Model 6"]),
                           regressor_order=[
                               # SEER
                               'VACSER1',                         
                               #Temperature Indirect Effect (Share of Days)
                               'Share',                          
                               #Cooling Infrastructure
                               'AC_units_3 or more',
                               'AC_units_Two','Central-Separate AC',
                               'Central-Unknown','VFANNUM',
                               #Housing Infrastructure
                               'Apartment/Condo/Townhouse','Mobile home',
                               'Sqft_1,500 - 2,999','Sqft_3,000 or more',
                               'VHOUSEHOLD','VRESAGE',
                               #Income
                               '$15,000 to $24,999',
                               '$25,000 to $34,999','$35,000 to $49,999','$50,000 to $74,999',
                               '$75,000 to $99,999','$100,000 to $149,999',
                               '$150,000 or more',
                               #Other Important Factor s
                               'Intercept','N','R-squared',
                               'R-squared Adj.'] )
    dfoutput.tables[0].to_excel(path+"/Results/Table3.xlsx")
    print(dfoutput)
Cooling_slopes_summary_table()


#-----------------------------------------------------------------
last20_accounts = data_processing_rule(X_models)[2]
X_last20 = X_models.loc[last20_accounts,:]
y_last20 = y.loc[last20_accounts]
testing_X = X_last20[["Intercept", "Share","VFANNUM","VACSER1"]]
testing_X = pd.concat([testing_X,
                    #Cooling Infrastructure),
                    pd.get_dummies(X_last20["VACUNITS"][X_last20["VACUNITS"]!= ""],
                                   prefix= "AC_units"),
                    pd.get_dummies(X_last20["VACTYPE"][( 
                        (X_last20["VACTYPE"]!="") &
                        (X_last20["VACTYPE"]!="Central-Gas"))]),
                    #Housing Infrastructure  
                    X_last20["VHOUSEHOLD"],
                    X_last20["VRESAGE"],
                    pd.get_dummies(X_last20["VBANSQFEET"][
                          X_last20["VBANSQFEET"] !=""], prefix= "Sqft"),
                     pd.get_dummies(X_last20["VBANDWELL"][
                          X_last20["VBANDWELL"] != ""])],axis=1)

testing_y = y_last20
[testing_y,testing_X ] = drop_empty_OLS_variables(testing_y , testing_X)

testing_X.drop(axis=1, labels=["AC_units_One","Central-Heat pump", 
                            "Single family home",
                            "Sqft_Less than 1,500"], inplace= True)

testing_t = pd.DataFrame(index= testing_X.index,
                         columns = training_X.columns)
testing_t.loc[:,testing_X.columns] = testing_X
testing_t.loc[:,~training_X.columns.isin(testing_X.columns)]= 0
tesing_predictions = training_model_robust.predict(
    testing_t.reset_index(drop= True))


#Predictive ability for the model's results 

testing_y_demo = add_race_income(pd.DataFrame(testing_y))

def SI_Table_cooling_model_prediction():
    from sklearn.metrics import mean_squared_error
    RMSE_in_sample = mean_squared_error(training_y,training_model_robust.predict())
    COVRMSE_in_sample = RMSE_in_sample/training_y.mean()*100
    RMSE_out_sample = mean_squared_error(testing_y,tesing_predictions)
    COVRMSE_out_sample = RMSE_out_sample/testing_y.mean()*100

    SI_cooling_slope_model_table = pd.DataFrame(data= [
        training_y.count(),training_y.mean(),RMSE_in_sample,COVRMSE_in_sample],
        columns = ["In-Sample"], dtype= float).T
    
    SI_cooling_slope_model_table.columns = ["Count", "Mean","RMSE","CV(RMSE)"] 
    
    SI_cooling_slope_model_table.loc["Out_of_Sample"] = [
        testing_y.count(),testing_y.mean(),RMSE_out_sample,COVRMSE_out_sample]
    SI_cooling_slope_model_table.to_excel(path+"/Results/SI_Figure5_table.xlsx")
    return(SI_cooling_slope_model_table)
SI_Table_cooling_model_prediction()

graph_vis_data = testing_y_demo.copy(deep=True)
graph_vis_data.rename(columns ={"17/18":"Actual"}, inplace= True)
graph_vis_data["Predictions"]= tesing_predictions.values
colors_= ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def SI_figure5():
    fig,axs = plt.subplots(3,1,dpi=600,sharey= True, sharex= True,figsize= (12,15),
                           constrained_layout= True)
    sns.scatterplot(ax=axs[0],y =  "Actual", data = graph_vis_data, s=50,
                 x ="Predictions", hue= "IG_num", style= "IG_num",  
                              x_jitter= True,edgecolor= 'k',
                 y_jitter= True)
    sns.scatterplot(ax=axs[1],y =  "Actual", 
                    data = graph_vis_data[graph_vis_data.Race != ''], s=50,
                 x ="Predictions",hue= "Race",style= "Race",  
                              x_jitter= True,edgecolor= 'k',
                 y_jitter= True)
    sns.scatterplot(ax=axs[2],y =  "Actual",
                    data = graph_vis_data[(graph_vis_data.Age!='')], s=50,
                 x ="Predictions", hue= "Age",style= "Age",  
                              x_jitter= True,edgecolor= 'k',
                 y_jitter= True)
    axs[0].legend(labels =income_group_labels, 
        title= "Income Groups",title_fontsize= "large",
               frameon=False, ncol=1, loc= 'lower right', fontsize=13)
    axs[1].legend( 
        title= "Race/Ethnicity",title_fontsize= "large",
               frameon=False, ncol=1, loc= 'lower right', fontsize=13)
    axs[2].legend( 
        title= "Age",title_fontsize= "large",
               frameon=False, ncol=1, loc= 'lower right', fontsize=13)
    fig.supylabel("Actual CDD Slopes",fontsize=20)
    fig.supxlabel("Predictions CDD Slopes",fontsize=20)
    for i in range(3):
        axs[i].set_xlim(0,5.5)
        axs[i].set_ylim(0,5.5)
        axs[i].set_xlabel("kWh/1$^\circ$F", fontsize=15)
        axs[i].set_ylabel("kWh/1$^\circ$F",fontsize=15)
        axs[i].plot(axs[i].get_xlim(), axs[i].get_ylim(), ls="--", c="black",
                    linewidth=1)
        sns.despine(ax=axs[i], top= True, right= True )
    plt.savefig(path+"/Results/SI_figure5.png",dpi=600)

    return()

SI_figure5()

