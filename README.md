# Climate-impacts-and-energy-burden

Pre-Processing scripts: 
  - **Data_processing.py** - uses the code to aggregate hourly data found in (Cong et al., 2022) to identify households with at least 360 days of daily observations, and fills missing household electricity prices with the average of the day before and after.  
   - **LOCA_data_processing.py** - uses downloaded daily average temperature maxs and mins to find the average daily temperauture for a centralized geographic area (this version of the code only includes 1 location point-estimate). 
   
Results scripts:
  - **temperature_response_model.py** - develops annual household level temperature response functions using total daily electricity smart meter data. This script includes the 5-parameter regression fixed-effects model. (Available upon reasonable request)
  - **Cooling_slope_reg_model.py** - modifies cooling slopes using the AC efficiency, share of days needed to cool, and infrastructure. (Available upon reasonable request)
  - **SEER_Calibration.py** - modifies SEER rating estimates for households that do not have a SEER rating (Available upon reasonable request)
  - **Future_consumption_simulations.py** - saves and creates climate electricity simulated consumption
  - **Future_consumption_simulations_new_betas.py** - saves and creates climate electricity simulated consumption with the new betas created using Cooling_slope_reg_model.py and SEER_Calibration.py

**Figures are generated from the following scripts:**
- Figure 2: 
- Figure 3:
- Figure 4:
- SI Figure D.1: 
- SI Figure E.1:


**Tables are generated from the following scripts:**

- SI Table D.1: 
- SI Table D.2:
- SI Table D.3: 
- SI Table E.1: 
- SI Table E.2:
- SI Table E.3:
- SI Table E.4:
- SI Table E.5:
- SI Table E.6:
- SI Table E.7: 


Dependencies:
  - Python: 3.9.7 
  - cython >=0.21                 :  0.29.32  
  - matplotlib >=2.0.0            :  3.5.3 
  - numpy >=1.7                   :  1.21.5 
  - pandas >=1.1.1                :  1.4.4 
  - scipy >=0.17.0                :  1.9.3 
  - sympy >=0.7.3                 :  1.11.1 
