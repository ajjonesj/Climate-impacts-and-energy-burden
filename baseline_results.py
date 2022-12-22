# -*- coding: utf-8 -*-
"""
This module performs the linear regression used in the main energy model.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from functions_and_labels import filtered_hh,colors_, add_race_income, ls_list, income_group_numbers, \
    median_CI_income_race_age

# Downloads the variables needed for the model
path = 'C:/Users/andre/Box/Andrew Jones_PhD Research/Climate change impacts on future electricity consumption and' \
       ' energy burden'
annual_cooling_points = pd.read_csv(path + '/Data/Baseline/Individual_Cooling_Balance_points.csv').set_index(
    "BILACCT_K")
annual_heating_points = pd.read_csv(path + '/Data/Baseline/Individual_Heating_Balance_points.csv').set_index(
    "BILACCT_K")
intercept_coef = pd.read_excel(path + '/Data/Baseline/2015_2019_intercept_coefficients.xlsx', header=0,
                               index_col=0)
CDD_coeff = pd.read_excel(path + '/Data/Baseline/2015_2019_CDD_coefficients.xlsx',
                          header=0,
                          index_col=0)
HDD_coeff = pd.read_excel(path + '/Data/Baseline/2015_2019_HDD_coefficients.xlsx',
                          header=0,
                          index_col=0)
panel_data = pd.read_csv(path + '/Data/Daily_consumption.csv', parse_dates=[0],
                         index_col=0)

exogen = pd.read_csv(path + '/Data/exogen_data_mod.csv', parse_dates=[0],
                     index_col=0)

avg_temps = exogen["temp_avg"].groupby(exogen.index.date).mean()
avg_temps.index = avg_temps.index.astype('datetime64[ns]')
household_CDD = pd.read_excel(path + '/Data/baseline/household_VBCDD.xlsx', index_col=[0])

models_2015_2019 = pd.read_pickle(path + '/Data/Baseline/2015_2019_regression_models.pkl')

survey_all = pd.read_stata(
    'C:/Users/andre/Box/RAPID COVID Residential Energy/Arizona_Full datasets/Arizona 2019-2020 dataset/RET_2017_part1.dta')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7, 4, 5, 6, 2, 1, 8, 3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"] != ""].unique()):
    survey_all.loc[survey_all["VINCOME"] == ig_num, ["VINCOME_new"]] = int(ig_unsorted[idxs])

survey_all.replace(
    {
        "65-74 yrs old": "65 yrs or older",
        "75+ yrs old": "65 yrs or older"}, inplace=True)
electric_households = survey_all["BILACCT_K"][~(
        (survey_all.VHEATEQUIP == "Separate gas furnace") |
        (survey_all.VHEATEQUIP == 'Gas furnace packaged with AC unit (sometimes called a gas pa)') |
        (survey_all.VACTYPE == 'AC unit packaged with gas heating (sometimes called a gas pa')
)].values

# %%Defines the variables needed for graph
annual_cooling_points_graph = add_race_income(
    annual_cooling_points.copy(deep=True))
annual_heating_points_graph = add_race_income(
    annual_heating_points.copy(deep=True))
comfort_zone_range = add_race_income(
    annual_cooling_points.sub(annual_heating_points,
                              axis="index",
                              fill_value=False))

CDD_coeff_add = add_race_income(CDD_coeff)
HDD_coeff_add = add_race_income(HDD_coeff)
intercept_coef_add = add_race_income(intercept_coef)

# %% ---------------Baseline Results setion----------------------
five_pt_coeft_1718 = pd.read_excel(path + '/Data/Baseline/2017_2018/2017_2018_coefficients.xlsx',
                                   index_col=(0))
five_pt_coeft_1718.drop(axis=0, labels=["BILACCT_K"], inplace=True)
temp_avgs = exogen.loc["05-01-2017":"04-30-2018"]["temp_avg"]
exogen_test = exogen.loc["05-01-2017":"04-30-2018"]

temp_avgs = temp_avgs.groupby(temp_avgs.index.date).mean()
exogen_test.loc[:, "VCDD"] = np.nan
exogen_test.loc[:, "VHDD"] = np.nan
temp_response_df = pd.DataFrame(index=temp_avgs.index,
                                columns=five_pt_coeft_1718.index)
for hh_count, house in enumerate(five_pt_coeft_1718.index):
    # Dataframe of the accounts in the income group
    daily_HDD_house = (annual_heating_points.loc[house, "17/18"] - temp_avgs).clip(lower=0)
    daily_CDD_house = temp_avgs.subtract(annual_cooling_points.loc[house, "17/18"]).clip(lower=0)
    temp_response_df.loc[:, house] = five_pt_coeft_1718.loc[house, "intercept"] + \
                                     five_pt_coeft_1718.loc[house, "HDD"] * daily_HDD_house + \
                                     five_pt_coeft_1718.loc[house, "CDD"] * daily_CDD_house
temp_response_df.dropna(axis=1, inplace=True)


def Figure_2():
    plt.rcParams["font.family"] = "Arial"
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    sns.color_palette("tab10")
    fig, axs = plt.subplots(figsize=(9, 5), dpi=600, sharex=True, sharey=True)
    ig_number_in = [1, 3, 6, 8]
    for idx, ig_numbers in enumerate(ig_number_in):
        accts_ig = survey_all["BILACCT_K"][survey_all.VINCOME_new == ig_numbers]
        ele_ig_acct = accts_ig[accts_ig.isin(electric_households)]
        income_data = temp_response_df.loc[:, temp_response_df.columns.isin(ele_ig_acct)]
        income_data = income_data.stack()
        income_data = income_data.reset_index()
        income_data = income_data.join(temp_avgs, on=["level_0"])
        income_data.columns = ["Date", "Accts", "kwh", "temp_avg"]
        sns.lineplot(data=income_data, x="temp_avg",
                     y="kwh", estimator=np.median, ci=95,
                     color=colors_[idx - 1], n_boot=5000,
                     seed=30, palette=("TABLEAU_COLORS"),
                     linewidth=2,
                     ls=ls_list[idx],
                     label=income_group_numbers[int(ig_number_in[idx]) - 1],
                     legend=None)

    axs.set_ylabel("")
    axs.set_xlabel("")
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)
    plt.suptitle("A. Income",
                 fontsize='25',
                 fontweight='bold')
    axs.legend(frameon=False, loc="upper left", fontsize=15)
    sns.despine(offset=0, trim=False)
    fig.supylabel("Simulated Daily Electricity Consumption\n (kWh per day)", fontsize=18)
    fig.supxlabel("Average Daily Outdoor Temperature ($^\circ$F)", fontsize=20)
    fig.tight_layout()
    plt.savefig(path + '/Results/trf_income.png', format='png', dpi=600)

    fig, axs = plt.subplots(figsize=(9, 5), dpi=600, sharex=True, sharey=True)
    races = ["Black or African American", "Hispanic", "White/Caucasian", "Asian"]
    for idx, races in enumerate(races):
        accts_race = survey_all["BILACCT_K"][survey_all.VETHNIC == races].values
        race_data = temp_response_df.loc[:, temp_response_df.columns.isin(accts_race)]
        race_data = race_data.stack()
        race_data = race_data.reset_index()
        race_data = race_data.join(temp_avgs, on=["level_0"])
        race_data.columns = ["Date", "Accts", "kwh", "temp_avg"]
        sns.lineplot(data=race_data, x="temp_avg",
                     y="kwh", estimator=np.median, ci=95,
                     palette="Set1_r", n_boot=1000,
                     seed=30,
                     linewidth=2.5, ls=ls_list[idx],
                     label=races)
    axs.set_ylabel("")
    axs.set_xlabel("")
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)
    axs.legend(frameon=False, loc="upper left", fontsize=15)
    sns.despine(offset=0, trim=False)
    plt.suptitle("B. Race",
                 fontsize='25',
                 fontweight='bold')
    fig.supylabel("Simulated Daily Electricity Consumption\n (kWh per day)", fontsize=18)
    fig.supxlabel("Average Daily Outdoor Temperature ($^\circ$F)", fontsize=20)
    fig.tight_layout()
    plt.savefig(path + '/Results/trf_race.png', format='png', dpi=600)

    survey_all.loc[:, "VHH_AGECODE_new"] = survey_all.VHH_AGECODE.replace(
        {'75+ yrs old': '65 yrs or older',
         "65-74 yrs old": "65 yrs or older"})
    age = ["65 yrs or older", '18-24 yrs old', '45-54 yrs old', '35-44 yrs old', ]
    sns.set_palette("nipy_spectral")
    fig, axs = plt.subplots(figsize=(9, 5),
                            dpi=600,
                            sharex=True,
                            sharey=True)
    for idx, age_code in enumerate(age):
        accts_age = survey_all["BILACCT_K"][survey_all.VHH_AGECODE_new == age_code].values
        age_data = temp_response_df.loc[:, temp_response_df.columns.isin(accts_age)]
        age_data = age_data.stack()
        age_data = age_data.reset_index()
        age_data = age_data.join(temp_avgs, on=["level_0"])
        age_data.columns = ["Date", "Accts", "kwh", "temp_avg"]
        sns.lineplot(data=age_data, x="temp_avg",
                     seed=30,
                     y="kwh", estimator=np.median, ci=95,
                     n_boot=1000, color=colors_[idx - 1],
                     linewidth=2.5, ls=ls_list[idx],
                     label=age_code)
    axs.set_ylabel("")
    axs.set_xlabel("")
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)
    axs.legend(frameon=False, loc="upper left", fontsize=15)
    sns.despine(offset=0, trim=False)
    plt.suptitle("C. Age",
                 fontsize='25',
                 fontweight='bold')
    fig.supylabel("Simulated Daily Electricity Consumption\n (kWh per day)", fontsize=18)
    fig.supxlabel("Average Daily Outdoor Temperature ($^\circ$F)", fontsize=20)
    fig.tight_layout()
    plt.savefig(path + '/Results/trf_age.png', format='png', dpi=600)


Figure_2()

temp_response_df_list = temp_response_df.melt(id_vars="Unnamed: 0",
                                              var_name = "BILACCT_K",
                                              value_name = "daily_kwh"
                                              )
temp_response_df_list.loc[:,"BILACCT_K"] = temp_response_df_list.BILACCT_K.astype("int32")
temp_response_df_list.join(survey_all.loc[:,["VINCOME","VETHNIC","VHH_AGECODE"]],
                           on=["BILACCT_K"])
temp_response_df.to_excel(path + '/Results/Simulated_daily_consumption_temp_only.xlsx')
temp_response_df = pd.read_excel(path + '/Results/Simulated_daily_consumption_temp_only.xlsx')

panel_data_summary = panel_data.loc["05-01-2017":"04-30-2018"].stack().reset_index()
panel_data_summary.columns = ["date_s", "BILACCT_K", "Daily_kWh"]
panel_data_summary.loc[:, "BILACCT_K"] = panel_data_summary.loc[:, "BILACCT_K"].astype(int)
panel_data_summary = panel_data_summary.set_index("BILACCT_K").join(
    intercept_coef_add.loc[:, ["17/18", "IG_num", "Income", "Age",
                               "AC_Type", "Sqft", "Race", "Occupancy"]],
    on="BILACCT_K")

panel_data_summary.to_csv(path + '/Data/Baseline_Summary_data.csv')

# %%Creates Bootstrapping Confidence interval
rng = np.random.default_rng()


#%% Table 2 analysis
def median_CI_table():
    baseload_data = intercept_coef_add.loc[:, ["17/18", "IG_num", "Income", "Age",
                                               "AC_Type", "Sqft", "Race", "Occupancy"]]

    CBP_data = annual_cooling_points_graph.loc[:, ["17/18", "IG_num", "Income",
                                                   "AC_Type", "Sqft", "Race", "Age",
                                                   "Occupancy"]]
    VCDD_data = CDD_coeff_add.loc[:, ["17/18", "IG_num", "Income",
                                      "AC_Type", "Sqft", "Race", "Age",
                                      "Occupancy"]]
    baseload_CI = pd.concat(median_CI_income_race_age(
        filtered_hh(baseload_data)), axis=0)
    baseload_CI.columns = baseload_CI.columns + "_BL"
    CBP_CI = pd.concat(median_CI_income_race_age(
        filtered_hh(CBP_data)), axis=0)
    CBP_CI.columns = CBP_CI.columns + "_CBP"
    VCDD_CI = pd.concat(median_CI_income_race_age(
        filtered_hh(VCDD_data)), axis=0)
    VCDD_CI.columns = VCDD_CI.columns + "_VCDD"
    counts = pd.concat([
        pd.Series(data=filtered_hh(baseload_data).reset_index().count()["BILACCT_K"]),
        filtered_hh(baseload_data).reset_index().groupby(["Age"]).count()["BILACCT_K"][1:],
        filtered_hh(baseload_data).reset_index().groupby(["IG_num"]).count()["BILACCT_K"],
        filtered_hh(baseload_data).reset_index().groupby(["Race"]).count()["BILACCT_K"].iloc[1:]],
        axis=0)
    Table2 = pd.concat([counts, baseload_CI, CBP_CI, VCDD_CI], axis=1).astype(float)
    pd.options.display.float_format = '{:,.1f}'.format

    Table2.rename(index={0: "All_households"}, inplace=True)
    Table2.rename(columns={0: "N"}, inplace=True)
    Table2.loc[:, "N"] = Table2.loc[:, "N"].astype(int)
    Table2.to_excel(path + '/Results/Table2.xlsx')
    return Table2

#Cooling Analysis
annual_cooling_points_demo = add_race_income(filtered_hh(annual_cooling_points))
def energy_equity_gap(annual_cooling_points_demo):
    
    #Energy Equity Gap (Income)
    EEG_IG = annual_cooling_points_demo.groupby(["IG_num"]).median().loc[:,"17/18"].max()-\
        annual_cooling_points_demo.groupby(["IG_num"]).median().loc[:,"17/18"].min()
    
    EEG_age = annual_cooling_points_demo.groupby(["Age"]).median().loc[:,"17/18"].max()-\
        annual_cooling_points_demo.groupby(["Age"]).median().loc[:,"17/18"].min()
    
    EEG_race = annual_cooling_points_demo[(annual_cooling_points.Race!="Pacific Islander")&
            (annual_cooling_points.Race!="Native Hawaiian or Other")].groupby(["Race"]).median().loc[:,"17/18"].max()-\
        annual_cooling_points_demo[(annual_cooling_points.Race!="Pacific Islander")&
                (annual_cooling_points.Race!="Native Hawaiian or Other")].groupby(["Race"]).median().loc[:,"17/18"].min()
    eeg_df = pd.DataFrame([EEG_IG,EEG_age,EEG_race], columns=["Energy Equity Gap (Max-Min)"],
            index =["Income", "Age", "Race"])
    return(eeg_df)
energy_equity_gap(annual_cooling_points_demo)

def summer_cooling_missed_days(annual_cooling_points_demo):
    #Dataframe with the average summertime temperatures for the baseline year (calendar year)
    avg_temps_1718_summer = avg_temps.loc["05-01-2017":"04-30-2018"][(
        (avg_temps.loc["05-01-2017":"04-30-2018"].index.month>=5) &
        (avg_temps.loc["05-01-2017":"04-30-2018"].index.month<=9))]
    #Missed days in the calendar year 
    days_missed_of_cooling_summer_all = pd.DataFrame([
        #Compares income groups 8 to income group 1 cooling balance point 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["IG_num"]).median()["17/18"].loc[8]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["IG_num"]).median()["17/18"].loc[1]].count(),
        #Compare black households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).median()["17/18"].loc["Black or African American"]].count(),
        #Compare hispanic households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).median()["17/18"].loc["Hispanic"]].count(),
        #Compare asian households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).median()["17/18"].loc["Asian"]].count(),
        #Compare native american households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).median()["17/18"].loc["American Indian/Alaska Native"]].count(),
            #Compare elderly households to mid-aged households 
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Age"]).median()["17/18"].loc["35-44 yrs old"]].count()-\
                avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                    ["Age"]).median()["17/18"].loc["65 yrs or older"]].count()
        ], columns = ["Missed Days"])
    days_missed_of_cooling_summer_all.index = ["Income Group 8 to 1",
                                       "White to Black", "White to Hispanic",
                                       "White to Aisan", 
                                       "White to American Indian",
                                       "Middle-aged to elderly"]
        
    days_missed_of_cooling_summer_worst_off = pd.DataFrame([
        #Compares income groups 8 to income group 1 cooling balance point 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["IG_num"]).median()["17/18"].loc[8]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["IG_num"]).quantile(q=0.9)["17/18"].loc[1]].count(),
        #Compare black households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).quantile(q=0.9)["17/18"].loc["Black or African American"]].count(),
        #Compare hispanic households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).quantile(q=0.9)["17/18"].loc["Hispanic"]].count(),
        #Compare asian households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).quantile(q=0.9)["17/18"].loc["Asian"]].count(),
        #Compare native american households to white households 
        avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
            ["Race"]).median()["17/18"].loc["White/Caucasian"]].count()-\
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Race"]).quantile(q=0.9)["17/18"].loc["American Indian/Alaska Native"]].count(),
            #Compare elderly households to mid-aged households 
            avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                ["Age"]).median()["17/18"].loc["35-44 yrs old"]].count()-\
                avg_temps_1718_summer[avg_temps_1718_summer>=annual_cooling_points_demo.groupby(
                    ["Age"]).quantile(q=0.9)["17/18"].loc["65 yrs or older"]].count()
        ], columns = ["Missed Days: Worst Off"])
    days_missed_of_cooling_summer_worst_off.index = ["Income Group 8 to 1",
                               "White to Black", "White to Hispanic",
                               "White to Aisan", 
                               "White to American Indian",
                               "Middle-aged to elderly"]

    missed_days_df = pd.concat([days_missed_of_cooling_summer_all,
                                days_missed_of_cooling_summer_worst_off],axis=1)
    return(missed_days_df)

summer_cooling_missed_days(annual_cooling_points_demo)
