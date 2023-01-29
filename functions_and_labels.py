# -*- coding: utf-8 -*-
"""
This script creates the functions that are called throughout the packages
@author: Andrew Jones
"""

import pandas as pd
import numpy as np

survey_all = pd.read_stata(
    'C:/Users/andre/Box/RAPID COVID Residential Energy/Arizona_Full datasets/Arizona 2019-2020 dataset/RET_2017_part1.dta')
survey_all["VINCOME_new"] = np.nan
ig_unsorted = [7, 4, 5, 6, 2, 1, 8, 3]
for idxs, ig_num in enumerate(survey_all["VINCOME"][survey_all["VINCOME"] != ""].unique()):
    survey_all.loc[survey_all["VINCOME"] == ig_num, ["VINCOME_new"]] = int(ig_unsorted[idxs])


def add_race_income(annual_cooling_points_graph):
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "Income"] = survey_all["VINCOME"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "Race"] = survey_all["VETHNIC"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "IG_num"] = survey_all["VINCOME_new"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "AC_Type"] = survey_all["VACTYPE"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "Sqft"] = survey_all["VSQFEET"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "Occupancy"] = survey_all["VHOUSEHOLD"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values
    annual_cooling_points_graph.loc[survey_all["BILACCT_K"][
                                        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)],
                                    "Age"] = survey_all["VHH_AGECODE"][
        survey_all["BILACCT_K"].isin(annual_cooling_points_graph.index)].values

    annual_cooling_points_graph.replace(
        {'Heat pump (same system heats and cools using electricity onl': 'Central-Heat pump',
         'AC unit packaged with gas heating (sometimes called a gas pa': 'Central-Gas',
         'Separate AC system that only cools': 'Central-Separate AC',
         "Don't know": "Central-Unknown",
         "65-74 yrs old": "65 yrs or older",
         "75+ yrs old": "65 yrs or older"}, inplace=True)
    return (annual_cooling_points_graph)


def labels():
    income_group_axis_labels = ["Income\n Group\n 1", "Income\n Group\n 2",
                                "Income\n Group\n 3", "Income\n Group\n 4", "Income\n Group\n 5", "Income\n Group\n 6",
                                "Income\n Group\n 7", "Income\n Group\n 8"]
    income_group_numbers = ["1. Less than 15,000 dollars",
                            "2. 15,000 to 24,999 dollars",
                            "3. 25,000 to 34,999 dollars",
                            "4. 35,000 to 49,999 dollars",
                            "5. 50,000 to 74,999 dollars",
                            "6. 75,000 to 99,999 dollars",
                            "7. 100,000 to 149,999 dollars",
                            "8. 150,000 dollars or more"]
    income_group_labels = ['Less than 15,000 dollars',
                           '15,000 to 24,999 dollars',
                           '25,000 to 34,999 dollars',
                           '35,000 to 49,999 dollars',
                           '50,000 to 74,999 dollars',
                           '75,000 to 99,999 dollars',
                           '100,000 to 149,999 dollars',
                           '150,000 or more dollars']

    income_group_legend_labels = ['$150,000 or more',
                                  '$100,000 to $149,999',
                                  '$75,000 to $99,999',
                                  '$50,000 to $74,999',
                                  '$35,000 to $49,999',
                                  '$25,000 to $34,999',
                                  '$15,000 to $24,999',
                                  'Less than $15,000']
    year_for_graphs = ["May 2015-April 2016",
                       "May 2016-April 2017",
                       "May 2017-April 2018",
                       "May 2018-April 2019"]
    return (
        income_group_axis_labels, income_group_numbers, income_group_labels, income_group_legend_labels,
        year_for_graphs)


def filtered_hh(data1):
    electric_households = survey_all["BILACCT_K"][~(
            (survey_all.VHEATEQUIP == "Separate gas furnace") |
            (survey_all.VHEATEQUIP == 'Gas furnace packaged with AC unit (sometimes called a gas pa)') |
            (survey_all.VACTYPE == 'AC unit packaged with gas heating (sometimes called a gas pa')
    )].values
    filtered_data = data1.loc[
                    data1.index.isin(electric_households), :]
    return (filtered_data)


from scipy.stats import bootstrap

rng = np.random.default_rng()


def median_CI_income_race_age(data_input, col_name):
    """This function finds the confidence interval for a series and outputs
    each income group's respective estimate and confidence interval
    
    Input: dataframe with values and the income group numbers"""
    # --------------------------------------------------------------------------
    # Income
    # --------------------------------------------------------------------------
    df_income = pd.DataFrame(index=np.arange(1, 9).astype(int),
                             columns=["Estimate", "LB", "UB"])
    for i in range(1, 9):
        data_est = data_input.loc[(data_input.IG_num == i), col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95, method='basic',
                        random_state=rng)
        df_income.loc[i, "Estimate"] = data_est.median()
        df_income.loc[i, ["LB", "UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Race
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_race = data_input[(
            (data_input.Race != "") &
            (data_input.Race != "Native Hawaiian or Other") &
            (data_input.Race != "Pacific Islander"))]
    # Defines a new dataframe for race estimates
    df_race = pd.DataFrame(index=data_input_race.Race.unique(),
                           columns=["Estimate", "LB", "UB"])
    for i, race in enumerate(df_race.index):
        data_est = data_input_race.loc[(data_input_race.Race == race),
                                       col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic',  # Why does basic work by not bias?? read up on this
                        random_state=rng)
        df_race.loc[race, "Estimate"] = data_est.median()
        df_race.loc[race, ["LB", "UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Age
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_age = data_input[
        (data_input.Age != "")]
    df_age = pd.DataFrame(index=data_input_age.Age.unique(),
                          columns=["Estimate", "LB", "UB"])
    for i, age in enumerate(df_age.index):
        data_est = data_input_age.loc[(data_input_age.Age == age),
                                      col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic',
                        random_state=rng)
        df_age.loc[age, "Estimate"] = data_est.median()
        df_age.loc[age, ["LB", "UB"]] = res.confidence_interval

    return df_income, df_race, df_age

def median_CI_income_race_age_sim(data_input, col_name):
    """This function finds the confidence interval for a series and outputs
    each income group's respective estimate and confidence interval
    
    Input: dataframe with values and the income group numbers"""
    # --------------------------------------------------------------------------
    # Income
    # --------------------------------------------------------------------------
    df_income = pd.DataFrame(index=np.arange(1, 9).astype(int),
                             columns=["Estimate", "LB", "UB"])
    for i in range(1, 9):
        data_est = data_input.loc[(data_input.IG_num == i), col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95, method='basic',
                        random_state=rng)
        df_income.loc[i, "Estimate"] = data_est.median()
        df_income.loc[i, ["LB", "UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Race
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_race = data_input[(
            (data_input.Race != "") &
            (data_input.Race != "Native Hawaiian or Other") &
            (data_input.Race != "Pacific Islander"))]
    # Defines a new dataframe for race estimates
    df_race = pd.DataFrame(index=data_input_race.Race.unique(),
                           columns=["Estimate", "LB", "UB"])
    for i, race in enumerate(df_race.index):
        data_est = data_input_race.loc[(data_input_race.Race == race),
                                       col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic',  # Why does basic work by not bias?? read up on this
                        random_state=rng)
        df_race.loc[race, "Estimate"] = data_est.median()
        df_race.loc[race, ["LB", "UB"]] = res.confidence_interval
    # --------------------------------------------------------------------------
    # Age
    # --------------------------------------------------------------------------
    # Removes empty data and households without less than 35 in sample size
    data_input_age = data_input[
        (data_input.Age != "")]
    df_age = pd.DataFrame(index=data_input_age.Age.unique(),
                          columns=["Estimate", "LB", "UB"])
    for i, age in enumerate(df_age.index):
        data_est = data_input_age.loc[(data_input_age.Age == age),
                                      col_name].dropna()
        boot_data = (data_est,)
        res = bootstrap(boot_data, np.median, confidence_level=0.95,
                        method='basic',
                        random_state=rng)
        df_age.loc[age, "Estimate"] = data_est.median()
        df_age.loc[age, ["LB", "UB"]] = res.confidence_interval

    return df_income, df_race, df_age


def add_demo_data_ele_sim(data_future_input, past_data_annual):
    """This function uses the axis of the dataframe indexed with acct, model, years, and
    decade information to add income, race, and age data"""

    for i, house in enumerate(past_data_annual.acct.unique()):
        # Income
        data_future_input.loc[data_future_input.acct == house,
                              "IG_num"] = survey_all["VINCOME_new"][survey_all[
                                                                        "BILACCT_K"] == house].values[0]
        # Race
        data_future_input.loc[data_future_input.acct == house,
                              "Race"] = survey_all["VETHNIC"][survey_all[
                                                                  "BILACCT_K"] == house].values[0]
        # Age
        data_future_input.loc[data_future_input.acct == house,
                              "Age"] = survey_all["VHH_AGECODE"][survey_all[
                                                                     "BILACCT_K"] == house].values[0]
        # AC Type
        data_future_input.loc[data_future_input.acct == house,
                              "AC_Type"] = survey_all["VACTYPE"][survey_all[
                                                                     "BILACCT_K"] == house].values[0]
        # Dwelling Size
        data_future_input.loc[data_future_input.acct == house,
                              "Sqft"] = survey_all["VSQFEET"][survey_all[
                                                                  "BILACCT_K"] == house].values[0]
        # Household Members
        data_future_input.loc[data_future_input.acct == house,
                              "Occupancy"] = survey_all["VHOUSEHOLD"][survey_all[
                                                                          "BILACCT_K"] == house].values[0]
        # AC Units
        data_future_input.loc[data_future_input.acct == house,
                              "ACUNITS"] = survey_all["VACUNITS"][survey_all[
                                                                      "BILACCT_K"] == house].values[0]
    data_future_input.replace(
        {"65-74 yrs old": "65 yrs or older",
         "75+ yrs old": "65 yrs or older"}, inplace=True)
    return (data_future_input)



colors_= ['C0', "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

ls_list = ['-', ':', '--', '-.']

income_group_numbers = ["1. Less than 15,000 dollars",
                        "2. 15,000 to 24,999 dollars",
                        "3. 25,000 to 34,999 dollars",
                        "4. 35,000 to 49,999 dollars",
                        "5. 50,000 to 74,999 dollars",
                        "6. 75,000 to 99,999 dollars",
                        "7. 100,000 to 149,999 dollars",
                        "8. 150,000 dollars or more"]
income_group_labels = ['Less than 15,000 dollars',
                       '15,000 to 24,999 dollars',
                       '25,000 to 34,999 dollars',
                       '35,000 to 49,999 dollars',
                       '50,000 to 74,999 dollars',
                       '75,000 to 99,999 dollars',
                       '100,000 to 149,999 dollars',
                       '150,000 or more dollars']

income_group_legend_labels = ['$150,000 or more',
                              '$100,000 to $149,999',
                              '$75,000 to $99,999',
                              '$50,000 to $74,999',
                              '$35,000 to $49,999',
                              '$25,000 to $34,999',
                              '$15,000 to $24,999',
                              'Less than $15,000']
year_for_graphs = ["May 2015-April 2016",
                   "May 2016-April 2017",
                   "May 2017-April 2018",
                   "May 2018-April 2019"]

ls_list = ['-', ':', '--', '-.']
marker_styles = ['+', 's', '.', 'v', '*', 'h', 'x', 'd']
