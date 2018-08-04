# CPI-Forecast
'''
Asset allocation system includes :
- Macro indicator forecast (GDP,CPI,etc), inflection point signal gereration, 
- back test, asset allocation, option pricing for leverage

First part:
Aimed at predicting CPI YoY ratio to provide signal of important inflection point for adjusting positions of allocated assets

We hope that the whole system is as automatic as possible.
Only some variables that change should be enter at the beginning part.
The part below includes the filter process for leading indicators
'''
Part1: China Food CPI MoM ratio forecast
# import data from China Wind Database
from WindPy import *
w.start()


# Changing Variables to be entered 
    start_friday       = '2007-01-05'
    start_date         = '2007-01-01'
    end_date           = '2018-07-11'
    poolcpi            = ["M0000706"]
    CPI                = 'CPI'
    chinese = ['猪肉','菜籽油','禽类','蔬菜类','小带鱼','苹果','大米','大带鱼']
    pool               = ['S0242966','S0242965','S0000240','S0000242','S0242992','S0242995','S0242961','S0242991']
    r1                 = -0.3 #treat the situation where CPI MoM changes fall slightly as uptrend
    r2                 = 0.3
    g1                 = 2
    g2                 = -2
    allshifts          = [0,1,2]#cpi a b
    oneshifts          = [0,1,2] #shift range for one indicator
    t_value            = 1.96
    max_shift_range    = 6 #max shift permisson in  "single_factor_optimal_shift"

# Constant
    changes = ['增', '减']#factor up/down
    modes=['增', '减'] #CPI YoY up/down


# Function definition 
    sublist : Choose n elements from a list.
    
    transfer: Change MoM ratio's frequency from week to month. Leading indicator's freqency is week, we need to change the high freqency data to match the dependent vairables.
    
    data_month : Create a dataframe to contain monthly data for both indicators and the dependent variable.
    calpct: Calculate percentage changes,taking shiftdic into consideration.
    
    predict_grade : Calculate data percentage changes match score,taking forecast data into consideration.
    
    score_turningpoint : Probability of success of important percentage changes.
    
    pct : Retern a boolean list of the up or down mode we need for the series.
    
    single_factor_optimal_shift : For single factor analysis to ensure the optimal leading period.
    
    单因子胜率 : Probability of success of single factor given different thresholds.
    
    predict_greade : Probability of success for forecast data when compared to the real data.
    
    granger : Granger test is applied to single factor analysis.
    
    shift_combinations : Each indicator has its own optimal leading period, if indicator a's optimal period is 2 months, we allow small fluctuation of the period's value. We will consider different cases of the periods (1,2,3 months).
    
    combination: Once given a list and different cases of the list, it will gernerate the different combinations.
    
    多因子目标频率胜率: Probability of success for multi-factors.
    
    多因子相关系数: Multi-factors'pair correlation values.
    
    多因子回归: Multi-factor's regression.
    
    各自单因子回归: Single factor regression.
    

# Single factor test
     test the correlation and optimal month ahead for those leading indicators
     test the significant level for each indicator with the dependent variable
     test the probabiltiy of success for each indicator to forecast the trend of the dependent variable
     
# Multi factor test : filter the best combination  
     pool = ['S0242966','S0242965','S0000240','S0000242','S0242992','S0242995','S0242961','S0242991']
     n: numbers of leading indicators to be chose from the indicator pool.
     i: the ith combination in the subpool
     we choose subpool = sublist(pool,5)[20]
     
# single and multi factor export 
    Regression and forecast within the sample (2017.1-2018.7)
    Use data within the sample for regression to predict the data out of the sample (2017.1-2017.6 to forecast 2017.7-2018.7)

# Rolling Regression and Forecast 
    Rule：Start on Dec,31st,2009, forecast next month's data (Jan,2010)，
    Real data is announced two month from now (2nd week in Feb,2010)

