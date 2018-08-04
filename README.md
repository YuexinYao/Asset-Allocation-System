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
    def sublist(inlist, n):
    return [list(x) for x in list(itertools.combinations(inlist, n))]

    '''Change MoM ratio's frequency from week to month'''
    def transfer(df): 
    plus1  = df.dropna().apply(lambda x: x[1]/100 + 1, axis=1)
    numb   = 1
    for i in list(plus1):
        numb   *= i
    return (numb-1)*100

    def data_month(data):
    temp_f  = data.loc[(t.date() for t in friday)].drop([CPI],axis=1)
    ym      = pd.DataFrame((t.strftime('%Y-%m-%d')[:7] for t in temp_f.index),columns=['ym'])
    ym.index = temp_f.index
    temp_df = pd.concat([ym,temp_f],axis=1)
    mo_list = []
    for i in temp_df.columns.values[1:]:
        temp = pd.concat([ym,temp_df[i]],axis=1)
        temp.columns = ['ym','1']
        mo = temp.dropna().groupby('ym').apply(transfer)
        mo_list.append(mo)
    quota_modf = pd.DataFrame(mo_list).T
    quota_modf.columns =temp_df.columns.values[1:]
    CPI0   = data[CPI].dropna() #CPI MoM month data
    ymcpi = pd.DataFrame((t.strftime('%Y-%m-%d')[:7] for t in CPI0.index),columns=['ymcpi'])
    ymcpi.index = CPI0.index
    CPI1  = pd.concat([ymcpi,CPI0],axis=1)
    CPI1.set_index(["ymcpi"],inplace=True)
    data_mo = pd.concat([CPI1[CPI],quota_modf],axis=1)
    return data_mo

    '''Calculate percentage changes,taking shiftdic into consideration'''
    def calpct(data_mo,shiftdic,factorsall):
        percentage=[]
        for factor in factorsall:
            calpct =[]
            for i in data_mo.index.values[1:]:#require data_mo's index to be numbers
                calpct.append((data_mo[factor].shift(shiftdic[factor])[i]-data_mo[factor].shift(shiftdic[factor])[i-1])/np.abs(data_mo[factor].shift(shiftdic[factor])[i-1]))
            percentage.append(calpct)
        df=pd.DataFrame(percentage).T
        df.columns = factorsall 
        temp=[]
        for i in data_mo.index.values[:-1]:
            temp.append(i+1)
        df.index = np.asarray(temp)
        return df

    '''Calculate data percentage changes match score,taking forecast data into consideration'''
    def predict_grade(predict0,col1,colp,shiftdic,r1,r2,p1,p2):#col1='rCPI' colp='pCPI'
        predict_pct = calpct(predict0[[col1,colp]],shiftdic,[col1,colp])
        grade=[]
        for i in predict_pct.index.values:
            if predict_pct[col1][i]>=r1 and predict_pct[colp][i]>=p1:
                grade.append(1)
            elif predict_pct[col1][i]<=r2 and predict_pct[colp][i]<=p2:
                grade.append(1)
            else:
                grade.append(0)
        g = sum(grade)/len(grade)
        return g

    def score_turningpoint(predict0,col1,col2,threshold1,threshold2):#col1 as turning point
        ratio = calpct(predict0[[col1,col2]],shiftdic,[col1,col2])
        score=[]
        for i in ratio.index.values:
            if ratio[col1][i]>=threshold1:
                if ratio[col2][i]>=0:
                    score.append(1)
                else:
                    score.append(0)
            elif ratio[col1][i]<=threshold2:
                if ratio[col2][i]<=0:
                    score.append(1)
                else:
                    score.append(0)
        return sum(score)/len(score)

    def pct(ser, grow_or_drop, b1,b2):
        if grow_or_drop == '增':
            return ((ser>0).shift(1).replace(False,-1)*ser.pct_change()).dropna()>=b1
        elif grow_or_drop == '减':
            return ((ser>0).shift(1).replace(False,-1)*ser.pct_change()).dropna()<=b2

    def single_factor_optimal_shift(data_mo):
        result = []
        maxcor = []
        for factor in factors:
            lag=[]
            cor=[]
            for i in range(max_shift_range):
                lag.append(i)
                cor.append(data_mo[CPI].corr(data_mo[factor].shift(i)))
            result.append(lag[cor.index(max(cor))])
            maxcor.append(round(max(cor),4))
        shiftcor = pd.DataFrame([result,maxcor])
        shiftcor.columns = factors
        return shiftcor.T


    def 单因子胜率(thresholds1,thresholds2):
        result = pd.DataFrame(index=['t-1', 't', 't+1'])
        for mode in modes: 
            for change in changes: #factors up/down dimension
                for factor in factors: 
                    wr_list_all_delays = []
                    t = single_factor_optimal_shift(data_mo)[0][factor].astype(int)
                    for delay in [t-1,t,t+1]: #factor在最优shift+/-1的shift
                        factor_change_when = delay + np.where(pct(data_mo[factor].shift(delay), change, thresholds1[factor],thresholds2[factor])==True)[0]+1
                        cpi_change_when = np.where(pct(data_mo[CPI], mode, thresholds1[CPI],thresholds2[CPI])==True)[0]+1
                        if len(factor_change_when) != 0:
                            wr = len(np.intersect1d(factor_change_when, cpi_change_when))/len(factor_change_when)
                            wr = round(wr,4)
                        else:
                            wr ='Nan'
                        wr_list_all_delays.append(wr)                    
                    result[factor+change+CPI+mode] = wr_list_all_delays
                    result = result.rename_axis('单因子胜率', axis=1)
        return result.T

    def calpct(data_mo,shiftdic,factorsall):
        percentage=[]
        for factor in factorsall:
            calpct =[]
            for i in data_mo.index.values[1:]:#要求index是数字
                calpct.append((data_mo[factor].shift(shiftdic[factor])[i]-data_mo[factor].shift(shiftdic[factor])[i-1])/np.abs(data_mo[factor].shift(shiftdic[factor])[i-1]))
            percentage.append(calpct)
        df=pd.DataFrame(percentage).T
        df.columns = factorsall 
        temp=[]
        for i in data_mo.index.values[:-1]:
            temp.append(i+1)
        df.index = np.asarray(temp)
        return df

    def predict_grade(predict0,col1,colp,shiftdic,r1,r2,p1,p2):#col1='rCPI' colp='pCPI'
        predict_pct = calpct(predict0[[col1,colp]],shiftdic,[col1,colp])
        grade=[]
        for i in predict_pct.index.values:
            if predict_pct[col1][i]>=r1 and predict_pct[colp][i]>=p1:
                grade.append(1)
            elif predict_pct[col1][i]<=r2 and predict_pct[colp][i]<=p2:
                grade.append(1)
            else:
                grade.append(0)
        g = sum(grade)/len(grade)
        return g


    def granger(data_mo):
        optimal_lag = []
        for factor in factors:
            if adfuller(data_mo[factor].dropna())[1]<0.1:
                granger_test_result= (grangercausalitytests(pd.concat([data_mo[CPI],data_mo[factor]],axis=1).dropna().values, 12, addconst=True, verbose=False))
                optimal_lag_temp   = 0
                F_test             = -1.0
                for key in granger_test_result.keys():
                    if granger_test_result[key][0]['params_ftest'][1]<0.05 :
                        _F_test_ = granger_test_result[key][0]['params_ftest'][0]
                        if _F_test_ > F_test:
                            F_test = _F_test_
                            optimal_lag_temp = key
                optimal_lag.append(optimal_lag_temp)
            else:
                optimal_lag.append('unstable')
        return optimal_lag


    def shift_combinations():
        cbn0 = []
        for factor in factors:
            t    = single_factor_optimal_shift(data_mo)[0][factor].astype(int)
            temp = [t-1,t,t+1]
            cbn0.append(temp)
        cbn = pd.DataFrame(list(itertools.product(*cbn0)), columns=factors)  
        names = []
        for i in range(len(cbn)):
            for factor in factors:
                names.append(factor+str(cbn[factor][i]))
        names = [names[x:x+len(factors)] for x in range(0, len(names), len(factors))]
        names = [', '.join(item) for item in names]  
        cbn.index = names
        return cbn

    def combinations(inlist):
        cbn = []
        for i in range(len(factors)):
            cbn.append(inlist)
        cbn = pd.DataFrame(list(itertools.product(*cbn)), columns=factors)  
        names = []
        for i in range(len(cbn)):
            for factor in factors:
                names.append(factor+str(cbn[factor][i]))
        names = [names[x:x+len(factors)] for x in range(0, len(names), len(factors))]
        names = [', '.join(item) for item in names]  
        cbn.index = names
        return cbn


    def 多因子目标频率胜率(thresholds1,thresholds2):
        result   = pd.Series()
        result1  = pd.Series()
        combination_change = combinations(changes)
        combination_delay  = combinations(oneshifts)
        for mode in modes:
            where_cpi = [np.where(pct(data_mo[CPI], mode, thresholds1[CPI],thresholds2[CPI])==True)[0]+1]
            for i in range(len(combination_delay)):
                #for j in range(len(combination_change)):
                for j in [0,len(combination_change)-1]:
                    where_factors=[]
                    for factor in factors:
                        where_factors = where_factors + [np.where(pct(data_mo[factor].shift(combination_delay[factor].iloc[i]), combination_change[factor][j], thresholds1[factor],thresholds2[factor])==True)[0]+1+(combination_delay[factor].iloc[i])]
                    if len(set.intersection(*map(set,where_factors))) !=0:
                        wr = len(set.intersection(*map(set,where_cpi+where_factors)))/len(set.intersection(*map(set,where_factors)))
                    else:
                        wr = 'Nan'
                    result ['<'+combination_delay.index[i]+','+combination_change.index[j]+'> '+CPI+mode] = wr
                    result1['<'+combination_delay.index[i]+','+combination_change.index[j]+'> '+CPI+mode] = combination_change.index[j]
                    result2 = pd.concat([result,result1],axis=1)
        return result2


    def 多因子相关系数(data_mo):
        result = pd.Series()
        df = pd.DataFrame()
        combination_delay =  shift_combinations()
        for i in range(len(combination_delay)):
            temp = data_mo.copy()
            for factor in factors:
                temp[factor]=temp[factor].shift(combination_delay[factor].iloc[i])
            cor_mat = temp.corr()
            for factor in factors:
                for j in factorsall:
                    result['<'+factor +'和'+ j+'>']=cor_mat[factor][j]
            df[i] = result
        df.columns = [combination_delay.index.values]      
        return df

    def 多因子回归():   
        combination =  combinations(oneshifts) 
        result = []
        for i in range(len(combination)):
            reg = pd.DataFrame(columns = ['coef', 't', 'p']) 
            cell = pd.DataFrame() 
            cell['CPI'] = pd.Series(data_mo['CPI'].values, name = 'CPI')
            for factor in factors:
                cell[factor] = data_mo[factor].shift(combination[factor].iloc[i]).values
            cell = cell.dropna(how='any')
            reg['coef'] = sa.OLS(cell['CPI'], cell[factors]).fit().params
            reg['t'] = sa.OLS(cell['CPI'], cell[factors]).fit().tvalues
            reg['p'] = sa.OLS(cell['CPI'], cell[factors]).fit().pvalues
            reg['R2'] = sa.OLS(cell['CPI'], cell[factors]).fit().rsquared
            reg['adjR']= sa.OLS(cell['CPI'], cell[factors]).fit().rsquared_adj
            reg = reg.rename_axis('%s, R2=%s'%(combination.index[i], sa.OLS(cell['CPI'], cell[factors]).fit().rsquared), axis=1)
            result.append(reg)
        return result
    #one factor regression
    def 各自单因子回归(data3,factors,bestaheaddict,dictxy):   #namedict is y:x to get specific factors group； dictxy is x:y
        result = pd.DataFrame()
        for factor in factors:
            reg1=pd.DataFrame()
            t = int(bestaheaddict[factor])
            for delay in [t]:#[t-1,t,t+1]:
                reg = pd.DataFrame(columns = ['coef', 't', 'p'])
                cell = pd.DataFrame()   
                y=dictxy[factor]
                cell[y] = pd.Series(data3[y].values, name = y)
                cell[factor] = data3[factor].shift(delay).values
                cell = cell.dropna(how='any')
                reg['coef']  = sa.OLS(cell[y], cell[factor]).fit().params
                reg['t']     = sa.OLS(cell[y], cell[factor]).fit().tvalues
                reg['p']     = sa.OLS(cell[y], cell[factor]).fit().pvalues
                reg['R2']    = sa.OLS(cell[y], cell[factor]).fit().rsquared
                reg['adjR']  = sa.OLS(cell[y], cell[factor]).fit().rsquared_adj
                reg['ahead'] = delay
                reg1 = reg1.append(reg)
            result = result.append(reg1)
        return result
        
#  Filter the best combination  
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

