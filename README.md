# Climate change impacts on future residential electricity consumption and energy burden

## Pre-Processing scripts:
  - **Data_processing.py** - uses the code to aggregate hourly data found in (Cong et al., 2022) to identify households with at least 360 days of daily observations, and fills missing household electricity prices with the average of the day before and after.  
   - **LOCA_data_processing.py** - uses downloaded daily average temperature maxs and mins to find the average daily temperauture for a centralized geographic area (this version of the code only includes 1 location point-estimate). 
   
## Results scripts:
  - **temperature_response_model.py** - develops annual household level temperature response functions using total daily electricity smart meter data. This script includes the 5-parameter regression fixed-effects model. (Available upon reasonable request). The script produces the following tables and figures:
    - SI Table D.1
    
  - **Baseline_results.py** - analyzes the baseline results to create tables and figures including the energy equity gap and  (Available upon reasonable request). The script produces the following tables and figures:
    - SI Table D.1
    - SI Table D.2
    - SI Figure D.1
    
    
  - **Cooling_slope_reg_model.py** - modifies cooling slopes using the AC efficiency, share of days needed to cool, and infrastructure. (Available upon reasonable request). The script produces the following tables and figures:
    - SI Table D.3
    - SI Figure E.1
    - SI Table E.1
  
  - **SEER_Calibration.py** - modifies SEER rating estimates for households that do not have a SEER rating (Available upon reasonable request). 
  - **Future_consumption_simulations.py** - saves and creates climate electricity simulated consumption.
  
  - **Monthly_Summertime_analysis.py** - developes the short-run effects of temperature change results and the quantile regression of the percentage points increases.The script produces the following tables and figures: 
      - Figure 2
      - SI Table E.2
      - SI Table E.3
      - SI Table E.4
      - SI Table E.5
      - SI Table E.6
      
  - **Future_consumption_simulations_new_betas.py** - saves and creates climate electricity simulated consumption with the new betas created using -"Cooling_slope_reg_model.py" and "SEER_Calibration.py" scripts. The script produces the following tables and figures:
    - Figure 3
 
- **Energy_Burden.py** - compares the energy burden across efficiency ratings and without upgrades.The script produces the following tables and figures:
  - Figure 4
 
