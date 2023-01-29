# -*- coding: utf-8 -*-
"""
This script 
@author: Andrew Jones
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from functions_and_labels import labels, filtered_hh, add_demo_data_ele_sim, survey_all

path = 'Insert the location of the project file'
future_monthly_data = pd.read_csv(path + r'/Data/Consumption_Simulations/Monthly_estimates.csv')
climate_simualtions_historical_1718 = pd.read_csv(path + '/Data/Consumption_Simulations/historical.csv',
                                                  date_parser=('date_s'), low_memory=False)
climate_simualtions_historical_1718.date_s = climate_simualtions_historical_1718.date_s.astype('datetime64[ns]')
[income_group_axis_labels, income_group_numbers, income_group_labels,
 income_group_legend_labels, year_for_graphs] = labels()

monthly_projections = future_monthly_data.copy(deep=True)
past_data_annual = climate_simualtions_historical_1718.groupby(["acct", "model"]).sum()

# %%LOAD DATA
summertime_per_change = pd.read_csv(path + '/Data/Future_Projections/summertime_per_change_estimates.csv')
summertime_monthly_future = pd.read_csv(path + '/Data/Future_Projections/summertime_consumption_estimates.csv')
# %% Calculates the percentage change for monthly consumption


# %% Summertime changes
""" In this sub-section define summertime consumption as the energy used
from May to September"""


# -------------------------------------------------------------------------------
# Future Summers changes
# -------------------------------------------------------------------------------
def summertime_consumption():
    summertime_monthly_future = monthly_projections.reset_index()
    summertime_monthly_future.rename(columns={"RCP_dates": "Month"}, inplace=True)
    # Creates a new column to determine if month is in the summer or not
    summertime_monthly_future.loc[((summertime_monthly_future["Month"] >= 5) &
                                   (summertime_monthly_future["Month"] <= 9)),
                                  "Summer"] = "Yes"
    summertime_monthly_future.loc[~((summertime_monthly_future["Month"] >= 5) &
                                    (summertime_monthly_future["Month"] <= 9)),
                                  "Summer"] = "No"
    # Groups and sums the column content per household, GCM, year, and decade
    summertime_monthly_future = summertime_monthly_future[
        summertime_monthly_future.Summer == "Yes"].groupby(
        ["acct", "model", "Year", "Decade"]).sum()

    # Drops the month and unnamed lables
    summertime_monthly_future.drop(axis=1, labels=["Month", "Unnamed: 0"],
                                   inplace=True)

    # -------------------------------------------------------------------------------
    # Past Summers changes
    # -------------------------------------------------------------------------------
    summertime_monthly_past = climate_simualtions_historical_1718.groupby(
        ["acct", "model", climate_simualtions_historical_1718.date_s.dt.month]).sum()

    summertime_monthly_past.rename_axis(index={"date_s": "Month"}, inplace=True)
    summertime_monthly_past.reset_index(inplace=True)
    summertime_monthly_past.loc[((summertime_monthly_past["Month"] >= 5) &
                                 (summertime_monthly_past["Month"] <= 9)),
                                "Summer"] = "Yes"
    summertime_monthly_past.loc[~((summertime_monthly_past["Month"] >= 5) &
                                  (summertime_monthly_past["Month"] <= 9)),
                                "Summer"] = "No"
    summertime_monthly_past = summertime_monthly_past[
        summertime_monthly_past.Summer == "Yes"].groupby(
        ["acct", "model"]).sum()
    summertime_monthly_past.drop(axis=1, labels=["Month", "Unnamed: 0"],
                                 inplace=True)

    # -----------------------------------------------------------------------------
    # Percentage Change
    # -----------------------------------------------------------------------------
    # Joins the future summer data frame with the past summer changes to duplicate past summer so I can divide
    # axis by axis
    joined_summertime_data = summertime_monthly_future.join(summertime_monthly_past,
                                                            on=["acct", "model"],
                                                            rsuffix="_past")

    summertime_monthly_past_new = joined_summertime_data.loc[:,
                                  ['kwh_RCP4.5_past', 'kwh_RCP8.5_past',
                                   'cost_4.5_past', 'cost_8.5_past',
                                   'cool_kwh_RCP4.5_past', 'cool_kwh_RCP8.5_past',
                                   'cool_cost_RCP4.5_past', 'cool_cost_RCP8.5_past']]

    # Renames the columns to match future df
    summertime_monthly_past_new.columns = summertime_monthly_past_new.columns.str.strip("_past")

    #
    col_names = ['kwh_RCP4.5', 'kwh_RCP8.5', 'cool_kwh_RCP4.5', 'cool_kwh_RCP8.5'
                 ]
    summertime_per_change = pd.DataFrame(index=summertime_monthly_future.index,
                                         columns=col_names)

    for col_num, col_name in enumerate(col_names):
        summertime_per_change.loc[:, col_name] = summertime_monthly_future.loc[:, col_name].divide(
            summertime_monthly_past_new.loc[:, col_name]).subtract(1).multiply(100)

    summertime_per_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    summertime_per_change.dropna(axis=0, inplace=True)

    summertime_consum_change = summertime_monthly_future.subtract(summertime_monthly_past_new)
    summertime_consum_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    summertime_consum_change.drop("index", axis=1, inplace=True)
    summertime_consum_change.dropna(axis=0, inplace=True)

    # adds demographic data to the household dataframe
    summertime_per_change = add_demo_data_ele_sim(data_future_input=summertime_per_change.reset_index(),
                                                  past_data_annual=past_data_annual.reset_index())
    summertime_monthly_future = add_demo_data_ele_sim(data_future_input=summertime_monthly_future.reset_index(),
                                                      past_data_annual=past_data_annual.reset_index())

    summertime_monthly_future["cool_MWh_RCP8.5"] = summertime_monthly_future["cool_kwh_RCP8.5"] / 1000
    summertime_monthly_future["cool_MWh_RCP4.5"] = summertime_monthly_future["cool_kwh_RCP4.5"] / 1000
    summertime_monthly_future["MWh_RCP8.5"] = summertime_monthly_future["kwh_RCP8.5"] / 1000
    summertime_monthly_future["MWh_RCP4.5"] = summertime_monthly_future["kwh_RCP8.5"] / 1000

    # Saves the new dataframes
    summertime_per_change.to_csv(path + '/Data/Future_Projections/summertime_per_change_estimates.csv')
    summertime_monthly_future.to_csv(path + '/Data/Future_Projections/summertime_consumption_estimates.csv')

    return (summertime_per_change, summertime_monthly_future)


[summertime_per_change, summertime_monthly_future] = summertime_consumption()

# Analysis-------------------------------------------------------------
annual_per_change = pd.read_csv(path + r'/Data/Consumption_Simulations/Annual_per_change_estimates.csv')
# Analysis for the annual consumption relative and percent changes
summertime_per_change.groupby(["model"]).median()["kwh_RCP4.5"].quantile(q=[0.2, 0.5, 0.8])
summertime_per_change.groupby(["model"]).median()["kwh_RCP8.5"].quantile(q=[0.2, 0.5, 0.8])
summertime_per_change.groupby(["model"]).median()["cool_kwh_RCP4.5"].quantile(q=[0.2, 0.5, 0.8])
summertime_per_change.groupby(["model"]).median()["cool_kwh_RCP8.5"].quantile(q=[0.2, 0.5, 0.8])
summertime_per_change[summertime_per_change.Decade != 2060].groupby(["model"]).median()["cool_kwh_RCP8.5"].quantile(
    q=[0.2, 0.5, 0.8]) / 2


def Decade_summertime_table():
    """This function outputs the data used to create SI table 4 and 5"""
    Table85_summer_cooling = pd.concat([
        summertime_per_change.groupby(["Decade", "IG_num"])["cool_kwh_RCP8.5"].median(),
        summertime_per_change.groupby(["Decade", "Race"])["cool_kwh_RCP8.5"].median(),
        summertime_per_change.groupby(["Decade", "Age", ])["cool_kwh_RCP8.5"].median()],
        axis=0)
    overall_85 = pd.DataFrame(
        summertime_per_change.groupby(["Decade"])["cool_kwh_RCP8.5"].median().round(1))
    overall_85.columns = ["Overall"]
    tab_count = pd.concat([
        summertime_per_change.groupby(["Decade", "IG_num"]).nunique()["acct"],
        summertime_per_change.groupby(["Decade", "Race"]).nunique()["acct"],
        summertime_per_change.groupby(["Decade", "Age", ]).nunique()["acct"]],
        axis=0)
    Table85_summer_cooling = pd.concat([tab_count, Table85_summer_cooling],
                                       axis=1)
    Table85_summer_cooling = Table85_summer_cooling.reset_index().set_index(["Decade"])
    Table85_summer_cooling.rename(columns={"IG_num": "Demo_data"}, inplace=True)
    Table85_summer_cooling = Table85_summer_cooling.join(overall_85)

    Table45_summer_cooling = pd.concat([
        summertime_per_change.groupby(["Decade", "IG_num"])["cool_kwh_RCP4.5"].median(),
        summertime_per_change.groupby(["Decade", "Race"])["cool_kwh_RCP4.5"].median(),
        summertime_per_change.groupby(["Decade", "Age", ])["cool_kwh_RCP4.5"].median()],
        axis=0)
    overall_45 = pd.DataFrame(
        summertime_per_change.groupby(["Decade"])["cool_kwh_RCP4.5"].median().round(1))
    overall_45.columns = ["Overall"]
    tab_count = pd.concat([
        summertime_per_change.groupby(["Decade", "IG_num"]).nunique()["acct"],
        summertime_per_change.groupby(["Decade", "Race"]).nunique()["acct"],
        summertime_per_change.groupby(["Decade", "Age", ]).nunique()["acct"]],
        axis=0)
    Table45_summer_cooling = pd.concat([tab_count, Table45_summer_cooling],
                                       axis=1)
    Table45_summer_cooling = Table45_summer_cooling.reset_index().set_index(["Decade"])
    Table45_summer_cooling.rename(columns={"IG_num": "Demo_data"}, inplace=True)
    Table45_summer_cooling = Table45_summer_cooling.join(overall_45)

    # Saves the new dataframes
    Table85_summer_cooling.to_excel(path + '/Results/Decade_estimates_85.xlsx')
    Table45_summer_cooling.to_excel(path + '/Results/Decade_estimates_45.xlsx')

    return (Table85_summer_cooling, Table45_summer_cooling)

Decade_summertime_table()

def Decade_summertime_table_kwh():
    """This function outputs the data used to create SI table 4 and 5"""
    Table85_summer_cooling = pd.concat([
        summertime_monthly_future.groupby(["Decade", "IG_num"])["cool_kwh_RCP8.5"].median(),
        summertime_monthly_future.groupby(["Decade", "Race"])["cool_kwh_RCP8.5"].median(),
        summertime_monthly_future.groupby(["Decade", "Age", ])["cool_kwh_RCP8.5"].median()],
        axis=0)
    overall_85 = pd.DataFrame(
        summertime_monthly_future.groupby(["Decade"])["cool_kwh_RCP8.5"].median().round(1))
    overall_85.columns = ["Overall"]
    tab_count = pd.concat([
        summertime_monthly_future.groupby(["Decade", "IG_num"]).nunique()["acct"],
        summertime_monthly_future.groupby(["Decade", "Race"]).nunique()["acct"],
        summertime_monthly_future.groupby(["Decade", "Age", ]).nunique()["acct"]],
        axis=0)
    Table85_summer_cooling = pd.concat([tab_count, Table85_summer_cooling],
                                       axis=1)
    Table85_summer_cooling = Table85_summer_cooling.reset_index().set_index(["Decade"])
    Table85_summer_cooling.rename(columns={"IG_num": "Demo_data"}, inplace=True)
    Table85_summer_cooling = Table85_summer_cooling.join(overall_85)

    Table45_summer_cooling = pd.concat([
        summertime_monthly_future.groupby(["Decade", "IG_num"])["cool_kwh_RCP4.5"].median(),
        summertime_monthly_future.groupby(["Decade", "Race"])["cool_kwh_RCP4.5"].median(),
        summertime_monthly_future.groupby(["Decade", "Age", ])["cool_kwh_RCP4.5"].median()],
        axis=0)
    overall_45 = pd.DataFrame(
        summertime_monthly_future.groupby(["Decade"])["cool_kwh_RCP4.5"].median().round(1))
    overall_45.columns = ["Overall"]
    tab_count = pd.concat([
        summertime_monthly_future.groupby(["Decade", "IG_num"]).nunique()["acct"],
        summertime_monthly_future.groupby(["Decade", "Race"]).nunique()["acct"],
        summertime_monthly_future.groupby(["Decade", "Age", ]).nunique()["acct"]],
        axis=0)
    Table45_summer_cooling = pd.concat([tab_count, Table45_summer_cooling],
                                       axis=1)
    Table45_summer_cooling = Table45_summer_cooling.reset_index().set_index(["Decade"])
    Table45_summer_cooling.rename(columns={"IG_num": "Demo_data"}, inplace=True)
    Table45_summer_cooling = Table45_summer_cooling.join(overall_45)

    # Saves the new dataframes
    Table85_summer_cooling.to_excel(path + '/Results/Decade_estimates_85_kwh.xlsx')
    Table45_summer_cooling.to_excel(path + '/Results/Decade_estimates_45_kwh.xlsx')

    return (Table85_summer_cooling, Table45_summer_cooling)

Decade_summertime_table_kwh()

# %% Percentage cahange and additional MWh figure
summertime_per_change =pd.read_csv(path + '/Data/Future_Projections/summertime_per_change_estimates.csv')
summertime_monthly_future =pd.read_csv(path + '/Data/Future_Projections/summertime_consumption_estimates.csv')

summertime_per_change["IG_num"] = summertime_per_change["IG_num"][~(
    summertime_per_change.IG_num.isnull())].astype(int).astype("category")

summertime_monthly_future["IG_num"] = summertime_monthly_future["IG_num"][~(
    summertime_monthly_future.IG_num.isnull())].astype(int).astype("category")
# %% Final Version of Percentage Change and consumption
import string

def Figure_4():
    fig, axs = plt.subplots(3, 2, dpi=600, figsize=(20, 25),
                            constrained_layout=True)
    axs = axs.ravel()
    sns.boxplot(
        ax=axs[0],
        data=summertime_per_change[(
                (summertime_per_change.Race != "Pacific Islander") |
                (summertime_per_change.Race != "American Indian/Alaska Native") |
                (summertime_per_change.Race != "Other")
            # & ~ (summertime_per_change.acct.isin(summertime_per_change_outlier_accts))
        )].groupby(
            ["acct", "Decade", "Race"]).median().reset_index(),
        x="cool_kwh_RCP8.5",
        y="Race",
        hue="Decade",
        hue_order=[2030, 2040, 2050, 2060],
        showmeans=True, whis=[10, 90],
        palette='Accent',
        order=['Black or African American',
               'White/Caucasian', 'Asian', 'Hispanic'],
        medianprops=dict(linestyle='-', linewidth=2.5, color="black"),
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markersize": "11", "markeredgecolor": "black"},
        # boxprops = dict(edgecolor = None, alpha = 0.65),
        # whiskerprops = dict(color="black"),
        capprops=dict(color="black"), showfliers=False
        # flierprops = dict(marker='o', markersize=2, color = "gray")
    )

    sns.boxplot(
        ax=axs[2],
        data=summertime_per_change.groupby(
            ["acct", "Decade", "IG_num"]).median().reset_index(),
        x="cool_kwh_RCP8.5", y="IG_num", hue="Decade",
        palette='Accent', whis=[10, 90],
        showmeans=True, hue_order=[2030, 2040, 2050, 2060],
        medianprops=dict(linestyle='-', linewidth=2.5, color="black"),
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markersize": "11", "markeredgecolor": "black"},
        showfliers=False,
        # flierprops = dict(marker='o', markersize=1.5, color = "gray"),
        capprops=dict(color="black")
    )
    sns.boxplot(
        ax=axs[4],
        data=summertime_per_change.groupby(
            ["acct", "Decade", "Age"]).median().reset_index(),
        order=['18-24 yrs old', '25-34 yrs old', '35-44 yrs old',
               '45-54 yrs old', '55-64 yrs old', '65 yrs or older'],
        palette='Accent', hue_order=[2030, 2040, 2050, 2060],
        x="cool_kwh_RCP8.5", y="Age", hue="Decade",
        medianprops=dict(linestyle='-', linewidth=2.5, color="black"),
        showmeans=True, showfliers=False, whis=[10, 90],
        meanprops={"marker": "o", "markerfacecolor": "white", "markersize": "11", "markeredgecolor": "black"},
        # flierprops = dict(marker='o', markersize=1.5, color = "gray"),
        capprops=dict(color="black")
    )

    sns.barplot(
        ax=axs[1],
        data=summertime_monthly_future[(
                (summertime_monthly_future.Race != "Pacific Islander") |
                (summertime_monthly_future.Race != "American Indian/Alaska Native") |
                (summertime_monthly_future.Race != "Other")
            # &
            # ~(summertime_monthly_future.index.isin(summertime_per_change_outlier_accts_data.index)
        )].groupby(
            ["acct", "Decade", "Race"]).median().reset_index(),
        units="acct", palette='Accent',
        order=['Black or African American',
               'White/Caucasian',
               'Asian', 'Hispanic'],
        hue="Decade", hue_order=[2030, 2040, 2050, 2060], y="Race",
        x='MWh_RCP8.5', estimator=np.median,
        n_boot=1000,
        edgecolor="black", errcolor='black', errwidth=1.5, capsize=.08)

    sns.barplot(
        ax=axs[3],
        data=summertime_monthly_future.groupby(
            ["acct", "Decade", "IG_num"]).median().reset_index(),
        hue="Decade", hue_order=[2030, 2040, 2050, 2060],
        y="IG_num", x='cool_MWh_RCP8.5', estimator=np.median,
        n_boot=1000,
        units="acct", palette='Accent',
        edgecolor="black", errcolor='black', errwidth=1.5, capsize=.08)

    sns.barplot(
        ax=axs[5],
        data=summertime_monthly_future.groupby(
            ["acct", "Decade", "Age", "model"]).median().reset_index(),
        hue="Decade", y="Age", x='cool_MWh_RCP8.5',
        hue_order=[2030, 2040, 2050, 2060], n_boot=1000,
        estimator=np.median, units="acct",
        palette='Accent',
        order=['18-24 yrs old', '25-34 yrs old', '35-44 yrs old',
               '45-54 yrs old', '55-64 yrs old', '65 yrs or older'],
        edgecolor="black", errcolor='black', errwidth=1.5, capsize=.08)

    sns.despine(top=True, right=True)

    title_labels = list(string.ascii_lowercase[:6])
    for i in range(6):
        axs[i].set_title(
            ("(" + title_labels[i] + ")"), loc='center',
            fontname="Arial", fontsize=25, fontweight="bold",
        )
        axs[i].set_xlabel("")
        labels = axs[i].get_yticklabels()
        
        if i == 2:
            axs[i].set_yticklabels(labels=income_group_numbers,
                                   fontsize=20)
        else:
            axs[i].set_yticklabels(labels=labels,
                                   fontsize=25)

    for i in [1, 3, 5]:

        axs[i].get_legend().remove()
        axs[i].set_xlim(0, 10)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].set_yticklabels([])
        axs[i].tick_params(axis='both', which='major', labelsize=25)
    for i in [0, 2, 4]:
        axs[i].get_legend().remove()
        axs[i].set_xlim(0, 130)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].tick_params(axis='both', which='major', labelsize=25)


    handles, labels = axs[5].get_legend_handles_labels()
    axs[4].set_xlabel("Percentage Change (%)", fontsize=28, fontweight='bold')
    axs[5].set_xlabel("Megawatt-hours (MWh)", fontsize=28, fontweight='bold')

    fig.legend(loc="lower center", handles=handles,
               ncol=4, fontsize=30,
               bbox_to_anchor=(0.6, -0.032),
               frameon=False)
    # fig.tight_layout()
    plt.savefig(path + '/Results/per_change_and_MWH.png', format='png',
                dpi=600,bbox_inches='tight')
    return ()


Figure_4()


# %% Analysis for percentage change and add. consumption
# ------------------------------------------------------------------------------
def quant_analysis():
    endo_income = summertime_per_change['cool_kwh_RCP8.5'].dropna()
    exog_var_income = pd.concat([
        pd.get_dummies(summertime_per_change['Sqft'][summertime_per_change.index.isin(endo_income.index)].dropna()),
        pd.get_dummies(summertime_per_change['IG_num'][summertime_per_change.index.isin(endo_income.index)].dropna()),
        pd.get_dummies(summertime_per_change['Decade'][summertime_per_change.index.isin(endo_income.index)].dropna()),
        pd.get_dummies(summertime_per_change['ACUNITS'][summertime_per_change.index.isin(endo_income.index)].dropna()),
        pd.get_dummies(summertime_per_change['AC_Type'][summertime_per_change.index.isin(endo_income.index)].dropna()),
        summertime_per_change['Occupancy'][summertime_per_change.index.isin(endo_income.index)].dropna()],
        axis=1).dropna()
    exog_var_income.loc[:, "intercept"] = 1
    endo_income = endo_income[endo_income.index.isin(exog_var_income.index)]

    exog_var_income.drop([8, "4,000 or more sq.", 2020,
                          'Heat pump (same system heats and cools using electricity onl',
                          "3 or more"], axis=1, inplace=True)

    quant_income_10 = sm.QuantReg(endo_income, exog_var_income).fit(q=0.1,
                                                                    cov_type='cluster',
                                                                    cov_kwds={'groups': summertime_per_change.acct[
                                                                        summertime_per_change.index.isin(
                                                                            exog_var_income.index)]})
    quant_income_50 = sm.QuantReg(endo_income, exog_var_income).fit(q=0.5,
                                                                    cov_type='cluster',
                                                                    cov_kwds={'groups': summertime_per_change.acct[
                                                                        summertime_per_change.index.isin(
                                                                            exog_var_income.index)]})
    quant_income_90 = sm.QuantReg(endo_income, exog_var_income).fit(q=0.9,
                                                                    cov_type='cluster',
                                                                    cov_kwds={'groups': summertime_per_change.acct[
                                                                        summertime_per_change.index.isin(
                                                                            exog_var_income.index)]})

    from statsmodels.iolib.summary2 import summary_col
    quant_income = summary_col([quant_income_10, quant_income_50, quant_income_90],
                               model_names=["10th %.tile", "50th %.tile",
                                            "90th %.tile"],
                               stars=False, 
                               info_dict={"N": lambda x: ("%#8i" % x.nobs),
                                          'Pseudo R-squared': lambda x: "%#8.3f" % x.prsquared},
                               float_format='%.2f', regressor_order=quant_income_50.params.index.tolist())

    quant_income.tables[0].to_excel(path + "/Results/SI_Quant_income.xlsx")

    # --------------------
    # Race
    # --------------------

    endo_race = summertime_per_change['cool_kwh_RCP8.5'].dropna()
    exog_var_race = pd.concat([
        pd.get_dummies(summertime_per_change['Sqft'][summertime_per_change.index.isin(endo_race.index)].dropna()),
        pd.get_dummies(summertime_per_change['Race'][summertime_per_change.index.isin(endo_race.index)].dropna()),
        pd.get_dummies(summertime_per_change['Decade'][summertime_per_change.index.isin(endo_race.index)].dropna()),
        pd.get_dummies(summertime_per_change['ACUNITS'][summertime_per_change.index.isin(endo_race.index)].dropna()),
        pd.get_dummies(summertime_per_change['AC_Type'][summertime_per_change.index.isin(endo_race.index)].dropna()),
        summertime_per_change['Occupancy'][summertime_per_change.index.isin(endo_race.index)].dropna()],
        axis=1).dropna()
    exog_var_race.loc[:, "constant"] = 1
    endo_race = endo_race[endo_race.index.isin(exog_var_race.index)]
    exog_var_race.drop(
        ["White/Caucasian", "4,000 or more sq.", 2020,
         'Heat pump (same system heats and cools using electricity onl',
         "3 or more"], axis=1, inplace=True)

    quant_race_10 = sm.QuantReg(endo_race, exog_var_race).fit(q=0.1,
                                                              cov_type='cluster',
                                                              cov_kwds={'groups': summertime_per_change.acct[
                                                                  summertime_per_change.index.isin(
                                                                      exog_var_race.index)]})

    quant_race_50 = sm.QuantReg(endo_race, exog_var_race).fit(q=0.5,
                                                              cov_type='cluster',
                                                              cov_kwds={'groups': summertime_per_change.acct[
                                                                  summertime_per_change.index.isin(
                                                                      exog_var_race.index)]})
    quant_race_90 = sm.QuantReg(endo_race, exog_var_race).fit(q=0.9,
                                                              cov_type='cluster',
                                                              cov_kwds={'groups': summertime_per_change.acct[
                                                                  summertime_per_change.index.isin(
                                                                      exog_var_race.index)]})
    quant_race = summary_col([quant_race_10, quant_race_50, quant_race_90],
                             info_dict={"N": lambda x: ("%#8i" % x.nobs),
                                        'Pseudo R-squared': lambda x: "%#8.3f" % x.prsquared},
                             stars=False,
                             model_names=["10th %.tile", "50th %.tile", "90th %.tile"],
                             float_format='%.2f')

    quant_race.tables[0].to_excel(path + "/Results/SI_Quant_race.xlsx")
    # --------------------
    # Age
    # --------------------
    summertime_per_change.replace([-np.inf, np.inf], np.nan, inplace=True)

    summertime_per_change.dropna(inplace=True)
    endo_age = summertime_per_change['cool_kwh_RCP8.5'].dropna()
    exog_var_age = pd.concat([
        pd.get_dummies(summertime_per_change['Sqft'][summertime_per_change.index.isin(endo_age.index)].dropna()),
        pd.get_dummies(summertime_per_change['Age'][summertime_per_change.index.isin(endo_age.index)].dropna()),
        pd.get_dummies(summertime_per_change['Decade'][summertime_per_change.index.isin(endo_age.index)].dropna()),
        pd.get_dummies(summertime_per_change['ACUNITS'][summertime_per_change.index.isin(endo_age.index)].dropna()),
        pd.get_dummies(summertime_per_change['AC_Type'][summertime_per_change.index.isin(endo_age.index)].dropna()),
        summertime_per_change['Occupancy'][summertime_per_change.index.isin(endo_age.index)].dropna()], axis=1).dropna()
    exog_var_age.loc[:, "constant"] = 1
    endo_age = endo_age[endo_age.index.isin(exog_var_age.index)]

    exog_var_age.drop(['45-54 yrs old', "4,000 or more sq.", 2020,
                       'Heat pump (same system heats and cools using electricity onl',
                       "3 or more"], axis=1, inplace=True)
    quant_age_10 = sm.QuantReg(endo_age, exog_var_age).fit(q=0.1,
                                                           cov_type='cluster',
                                                           cov_kwds={'groups': summertime_per_change.acct[
                                                               summertime_per_change.index.isin(
                                                                   exog_var_age.index)]})
    quant_age_50 = sm.QuantReg(endo_age, exog_var_age).fit(q=0.5,
                                                           cov_type='cluster',
                                                           cov_kwds={'groups': summertime_per_change.acct[
                                                               summertime_per_change.index.isin(
                                                                   exog_var_age.index)]})
    quant_age_90 = sm.QuantReg(endo_age, exog_var_age).fit(q=0.9,
                                                           cov_type='cluster',
                                                           cov_kwds={'groups': summertime_per_change.acct[
                                                               summertime_per_change.index.isin(
                                                                   exog_var_age.index)]})
    quant_age = summary_col([quant_age_10, quant_age_50, quant_age_90],
                            model_names=["10th %.tile", "50th %.tile",
                                         "90th %.tile"],
                            stars=False,
                            info_dict={"N": lambda x: ("%#8i" % x.nobs),
                                       'Pseudo R-squared': lambda x: "%#8.3f" % x.prsquared},
                            float_format='%.2f', regressor_order=quant_race_50.params.index.tolist())

    quant_age.tables[0].to_excel(path + "/Results/SI_Quant_age.xlsx")
    return (quant_age, quant_income, quant_race)

[quant_age, quant_income, quant_race] = quant_analysis()
