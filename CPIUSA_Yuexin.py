# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 23:54:11 2018

@author: Yuexin Yao
"""

import numpy as np
import pandas as pd
from WindPy import *
w.start()
import statsmodels.api as sm
import itertools
import statsmodels.api as sa
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
%matplotlib qt5

######################  Data Processing ##############
tongbi = w.edb("G0000067,M0001227,G0000027,G0002410,G1156652, \
              G1156612,G0002417", "2000-01-01", "2018-07-04")
              # M0000005 ：期货结算价(连续):WTI原油; S5112405:现货离岸价	、
              # G0000067 : 美国:失业率:季调	
              # M0001227 : PPI:全部工业品:当月同比	
              # G0000027 : 美国:CPI:当月同比
              # G0002410 : 美国:CPI:能源：当月同比
              # G1156652 ：美国:CPI:服务,不含能源服务:季调
              # G1156612 ：美国:CPI:商品,不含食品和能源类商品:季调
              # G0002417 ：美国:CPI:食品：当月同比
ridu =  w.edb("S0031506, S0031645 ,M0000005", "2000-01-01", "2018-07-04")
              # S0031506 : CRB现货指数:食品	
              # S0031645 : 伦敦现货黄金:以美元计价
              #M0000005 ：期货结算价(连续):WTI原油;
              
ri =  w.edb("S0031506,M0000005", "2000-01-01", "2018-07-04","Fill=Previous")
ri1 = pd.DataFrame(ri.Data).T
ri1['time'] =  pd.to_datetime(ri.Times)
ri1.index=ri1['time']
wednesday            = pd.date_range(start = "2000-01-05", end =  "2018-07-04", freq = 'W-WED')
temp_f  = ri1.loc[(t.date() for t in wednesday)]
ym      = pd.DataFrame((t.strftime('%Y-%m-%d')[:7] for t in temp_f.index),columns=['ym'])
ym.index = temp_f.index
weekdf = pd.concat([ym,temp_f],axis=1)
weekdf=weekdf.rename(columns={0:'CRB_food',1:'WTI'})
weekdf['CRB_food']=100*(weekdf['CRB_food']-weekdf['CRB_food'].shift(4))/weekdf['CRB_food'].shift(4)
weekdf['WTI']=100*(weekdf['WTI']-weekdf['WTI'].shift(4))/weekdf['WTI'].shift(4)
weekdf.index=weekdf['ym']

pp = w.edb("M0001227 ", "2000-01-01", "2018-07-04","Fill=Previous")
pp1 = pd.DataFrame(pp.Data).T
pp1['time'] =  pd.to_datetime(pp.Times)
pp1['ym'] =pd.DataFrame(t.strftime('%Y-%m-%d')[:7] for t in pp1['time'])
pp1=pp1.rename(columns={0:'PPI'})
pp1.index=pp1['ym']

tongbi1 = pd.DataFrame(tongbi.Data).T
tongbi1['time'] = tongbi.Times
ridu1 = pd.DataFrame(ridu.Data).T
ridu1['time'] =  pd.to_datetime(ridu.Times)
tick = pd.date_range(start = '2000-01-31', end = '2018-07-04', freq = 'M')
tick = [t.date() for t in tick]

ridu1.index = ridu1['time']
dfg = ridu1.groupby(pd.TimeGrouper('M'))
business_end_day = dfg.agg({'time': np.max})['time'].tolist()#!
ridu2 = ridu1[ridu1['time'].isin(business_end_day)][:-1]#!
ridu2['time'] = tick
data2 = tongbi1.merge(ridu2, on = 'time')

data2 = data2.rename(columns = { '0_x':'unemployment', '1_x':'PPI', '0_y':'CRB_food', '2_x':'CPI',\
                               3:'CPI_nenergy', 4:'CPI_service', 5:'CPI_commodity', 6:'CPI_food','1_y':'gold','2_y':'WTI'})

#change to YoY
data2['WTI'] = (data2['WTI']-data2['WTI'].shift(12))/np.abs(data2['WTI'].shift(12))
data2['unemployment'] = (data2['unemployment']-data2['unemployment'].shift(12))/np.abs(data2['unemployment'].shift(12)) 
data2['PPI'] = data2['PPI']/100
data2['CRB_food'] = (data2['CRB_food']-data2['CRB_food'].shift(12))/np.abs(data2['CRB_food'].shift(12))
data2['CPI'] = data2['CPI'] /100
data2['CPI_nenergy'] = data2['CPI_nenergy']/100
data2['CPI_service'] = (data2['CPI_service']-data2['CPI_service'].shift(12))/np.abs(data2['CPI_service'].shift(12))
data2['CPI_commodity'] = (data2['CPI_commodity']-data2['CPI_commodity'].shift(12))/np.abs(data2['CPI_commodity'].shift(12))
data2['CPI_food'] = data2['CPI_food']/100

data2 = data2.dropna()
data2[data2==0] = np.nan
data2 = data2.fillna(method = 'bfill')#！

######################  Function Deffination ##############
def pct(ser, grow_or_drop, b1,b2):
    if grow_or_drop == '增':
        return ((ser>0).shift(1).replace(False,-1)*ser.pct_change()).dropna()>=b1
    elif grow_or_drop == '减':
        return ((ser>0).shift(1).replace(False,-1)*ser.pct_change()).dropna()<=b2
    
def 美国单因子胜率(thresholds1,thresholds2):
    result = pd.DataFrame(index=['t-3','t-2','t-1', 't', 't+1','t+2','t+3'])
    for mode in modes: 
        for change in changes: #因子上升、下降维度
            for factor in factors: 
                wr_list_all_delays = []
                t = aheaddic[factor].astype(int)
                for delay in [t-3,t-2,t-1,t,t+1,t+2,t+3]: #factor在最优shift+/-1的shift
                    factor_change_when = delay + np.where(pct(data3[factor].shift(delay), change, thresholds1[factor],thresholds2[factor])==True)[0]+1
                    cpi_change_when    = np.where(pct(data3[dictnew[factor]], mode, thresholds1[dictnew[factor]],thresholds2[dictnew[factor]])==True)[0]+1
                    if len(factor_change_when) != 0:
                        wr = len(np.intersect1d(factor_change_when, cpi_change_when))/len(factor_change_when)
                        wr = round(wr,4)
                    else:
                        wr ='Nan'
                    wr_list_all_delays.append(wr)                    
                result[factor+change+dictnew[factor]+mode] = wr_list_all_delays
                result = result.rename_axis('单因子胜率', axis=1)
    return result.T
def dicshift_combinations(bestaheaddic,factors):
    cbn0 = []
    for factor in factors:
        t    = int(bestaheaddic[factor])#
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

def 有常项多因子回归(fac,data_mo,yname,bestaheaddict):   
    combination = dicshift_combinations(bestaheaddict,fac)
    result = []
    for i in range(len(combination)):
        reg = pd.DataFrame(columns = ['coef', 't', 'p']) #单种组合回归结果dataframe
        cell = pd.DataFrame() #单种组合回归数据dataframe
        cell[yname] = pd.Series(data_mo[yname].values, name = yname)
        for factor in fac:
            cell[factor] = data_mo[factor].shift(combination[factor].iloc[i]).values
        cell = cell.dropna(how='any')
        est = sa.OLS(cell[yname], sm.add_constant(cell[fac])).fit()
        reg['coef'] = est.params
        reg['t'] = est.tvalues
        reg['p'] = est.pvalues
        reg['R2'] = round(est.rsquared,4)
        reg['adjR']= round(est.rsquared_adj,4)
        reg = reg.rename_axis('%s, R2=%s'%(combination.index[i],est.rsquared), axis=1)
        result.append(reg)
    return result

def 美国单因子回归():   
    result = pd.DataFrame()
    for factor in factors:
        reg1=pd.DataFrame()
        t = int(bestaheaddic[factor])
        for delay in [t-1,t,t+1]:
            reg = pd.DataFrame(columns = ['coef', 't', 'p']) #单种组合回归结果dataframe
            cell = pd.DataFrame() #单种组合回归数据dataframe   
            y=dictnew[factor]
            cell[dictnew[factor]] = pd.Series(data3[dictnew[factor]].values, name = dictnew[factor])
            cell[factor] = data3[factor].shift(delay).values
            cell = cell.dropna(how='any')
            est = sa.OLS(cell[y], sm.add_constant(cell[factor])).fit()
            reg['coef'] = est.params
            reg['t'] = est.tvalues
            reg['p'] = est.pvalues
            reg['R2'] = round(est.rsquared,4)
            reg['adjR']= round(est.rsquared_adj,4)
            reg['ahead'] = delay
            reg1 = reg1.append(reg)
        result = result.append(reg1)
    return result


def 美国多因子回归():   
    combination =  dicshift_combinations(bestaheaddic,factors)
    result = []
    for i in range(len(combination)):
        reg = pd.DataFrame(columns = ['coef', 't', 'p']) #单种组合回归结果dataframe
        cell = pd.DataFrame() #单种组合回归数据dataframe
        cell['CPI'] = pd.Series(data3['CPI'].values, name = 'CPI')
        for factor in factors:
            cell[factor] = data3[factor].shift(combination[factor].iloc[i]).values
        cell = cell.dropna(how='any')
        y='CPI'
        est = sa.OLS(cell[y], sm.add_constant(cell[factors])).fit()
        reg['coef'] = est.params
        reg['t'] = est.tvalues
        reg['p'] = est.pvalues
        reg['R2'] = round(est.rsquared,4)
        reg['adjR']= round(est.rsquared_adj,4)
        reg = reg.rename_axis('%s, R2=%s'%(combination.index[i], sa.OLS(cell[y], sm.add_constant(cell[factor])).fit().rsquared), axis=1)
        result.append(reg)
    return result

def sublist(inlist, n):
    return [list(x) for x in list(itertools.combinations(inlist, n))]


######################  Analysis ##############
shuchu1 = pd.DataFrame(np.zeros((4,2)), columns = ['ahead', 'correlation'], index = ['WTI', 'unemployment', 'PPI', 'CRB_food']) # 最大相关
shuchu2 = dict() # 所有相关系数
suoyin = dict({'WTI':'CPI_nenergy','unemployment': 'CPI_service', 'PPI':'CPI_commodity', 'CRB_food':'CPI_food'})
for col in ['WTI', 'unemployment', 'PPI', 'CRB_food']:
    col1 = col
    col2 = suoyin[col1]
    shuchu3 = pd.DataFrame(np.zeros((18,2)), columns = ['ahead', 'correlation']) # 单指标与CPI相关系数
    data4 = data2[[col1, col2, 'time']]#diff data5
    data4 = data4.dropna()
    for j in range(18):
        data4[col2].corr(data4[col1].shift(j))
        r = pd.Series.corr(data4[col1].shift(j), data4[col2][:-j])  if j !=0 \
            else pd.Series.corr(data4[col1], data4[col2])
        shuchu3.iloc[j] = (j, r)
    shuchu4 = abs(shuchu3)
    shuchu1.loc[col1] = shuchu3.iloc[shuchu4['correlation'].idxmax()]
    shuchu2[col1] = shuchu3
print('领先期数', shuchu1)

data3 = data2.copy()


dictnew = dict({'WTI':'CPI_nenergy','unemployment': 'CPI_service', 'PPI':'CPI_commodity', 'CRB_food':'CPI_food'})
factors = list(dictnew.keys())
factorsall = list(list(dictnew.values()))+list(dictnew.keys())+['CPI']
ahead   = list(shuchu1['ahead'].values)
aheaddic = dict(zip(factors,ahead))
thre_abs = dict.fromkeys(factorsall, 0)
changes = ['增', '减']#因子变化
modes=['增', '减']


gradeabs = 美国单因子胜率(thre_abs,thre_abs)

bestaheaddic = {'WTI': 1.0, 'unemployment': 3.0, 'PPI': 17.0, 'CRB_food': 7.0}


美国单因子回归()[['R2','adjR','ahead']]

factors
t_value            = 1.96
qq=pd.DataFrame()
pool = list(dictnew.keys())
for n in range(1,len(pool)+1):
    toutput =pd.DataFrame()
    for i in range(len(sublist(pool,n))):
        factors =sublist(pool,n)[i]
        multi_reg = 美国多因子回归()
        ttest0 = pd.DataFrame()
        for k in range(len(multi_reg)):
            ttest0[k] = multi_reg[k]['t']
            ttest0.loc['R2',k] = multi_reg[k]['R2'][0]
            ttest0.loc['adjR',k] = multi_reg[k]['adjR'][0]
        ttest0.columns =  [dicshift_combinations(bestaheaddic,factors).index.values]    
        ttest = ttest0.T.loc[(ttest0.T.loc[:,factors]>=t_value).replace(False, np.nan).dropna(how='any').index]
        ttest.insert(0,'组合i',[i]*len(ttest))
        toutput =toutput.append(ttest)
    toutput.insert(0,'指标数n',[n]*len(toutput))
    qq = qq.append(toutput)
qq.sort_values(by='adjR',ascending=False)

CPI ='CPI'
def 美国多因子目标频率胜率(data_mo,thresholds1,thresholds2):
    result   = pd.Series()
    result1  = pd.Series()
    combination_change = combinations(changes)
    combination_delay  = dicshift_combinations(bestaheaddic)
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
美国多因子目标频率胜率(data_mo,thre_abs,thre_abs)

def calpct(data_mo,shiftdic,factorsall):
    percentage=[]
    for factor in factorsall:
        calpct =[]
        for i in data_mo.index.values[1:]:#dataframe's index should be numbers
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

def score_turningpoint(predict0,col1,col2,threshold1,threshold2):#以col1识别拐点
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


bestaheaddict1={'WTI': 1.0, 'PPI': 16.0, 'CRB_food': 6.0}
factors =list(bestaheaddict1.keys())
comb               = dicshift_combinations(bestaheaddict1,factors)

data_all           = data3.copy()
ym                 = pd.DataFrame((t.strftime('%Y-%m-%d')[:7] for t in data_all['time']),columns=['ym'])
data_all           = data_all.reset_index(drop=True)
data_all['ym']     = pd.Series(ym['ym'].values,name='ym')
data_all.index     = data_all['ym'].values
k                  = int(np.where(data_all.index.values=='2017-06')[0])
data_mo            = data_all.iloc[:k]
data_mop           = data_all.iloc[k:]
i                  = int(np.where(comb.index=='WTI1, PPI16, CRB_food6')[0])
multi_coef         = 美国多因子回归()[i]['coef']
coef1=multi_coef[factors]

predict = pd.DataFrame()
data_predict = pd.DataFrame()
for factor in factors:
    data_predict[factor] = data_all[factor].shift(comb[factor].iloc[i]).iloc[-(len(data_mop)):].values
predict['date'] = data_mop.index.values
predict['rCPI'] =pd.Series(data_mop['CPI'].values, name = 'rCPI')
temp_mat = coef1.T*data_predict.iloc[:,0:4]
predict['pCPI'] = pd.Series(temp_mat.sum(axis=1).values+multi_coef['const'], name = 'pCPI')

col1='rCPI'
colp = 'pCPI'
r1=-0.005
r2=0.005
g1=0.02
g2=0.02
shiftdic = dict(zip(factors,comb.iloc[i].values));shiftdic['rCPI'] = shiftdic['pCPI'] = 0
score_r_type2   = predict_grade(predict,col1,colp,shiftdic,r1,r2,0,0)
score_abs_type2 = predict_grade(predict,col1,colp,shiftdic,0,0,0,0)
score_tp_type2  = score_turningpoint(predict,colp,col1,g1,g2)
score_tr_type2  = score_turningpoint(predict,col1,colp,g1,g2)
score_type2     = [score_r_type2,score_abs_type2,score_tp_type2,score_tr_type2]

fig, ax = plt.subplots(figsize=(8,6))
ax.plot( predict['rCPI'], 'b--', label='data')
ax.plot(predict['pCPI'], 'r--.',label='OLS')
ax.legend(loc='best')
plt.title('2007-2017.6 data to predict 2017.7-2018.7 WTI1, PPI16, CRB_food6;',fontsize = 20)

############################### Roll Forecast (Freqency:Month) #######################################
#规则：Today is Dec,31st,2009.Forecast CPi YoY of Jan,2010, which will be announced in Feb,2010
start_date         = '2007-01-01'
end_date           = '2018-08-01'

bestaheaddict1={'WTI': 1.0, 'PPI': 16.0, 'CRB_food': 6.0}
factors =list(bestaheaddict1.keys())
comb               = dicshift_combinations(bestaheaddict1,factors)
data_all

i                  = int(np.where(comb.index=='WTI1, PPI16, CRB_food6')[0])


k1                  = int(np.where(data_all.index=='2008-01')[0])#36
k2                 = int(np.where(data_all.index=='2018-06')[0])#138 201807
roll_predict = pd.DataFrame()
for k in range(k1,k2+1):
    data_mo            = data_all.iloc[:k]
    data_mop           = data_all.iloc[k:][:1]
    
    predict = pd.DataFrame()
    data_predict = pd.DataFrame()
    for factor in factors:
        data_predict[factor] = data_all[factor].shift(comb[factor].iloc[i]).iloc[k:][:1].values
    multi_coef = pd.DataFrame(columns = ['coef']) 
    cell = pd.DataFrame() 
    cell['CPI'] = pd.Series(data_mo['CPI'].values, name = 'CPI')
    for factor in factors:
        cell[factor] = data_mo[factor].shift(comb[factor].iloc[i]).values
    cell = cell.dropna(how='any')
    est = sa.OLS(cell['CPI'], sm.add_constant(cell[factors])).fit()
    multi_coef = est.params
    coef1=multi_coef[factors]

    predict['date'] = data_mop.index.values
    predict['rCPI'] =pd.Series(data_mop['CPI'].values, name = 'rCPI')
    temp_mat        = coef1*data_predict.iloc[:,0:6]
    predict['pCPI'] = pd.Series(temp_mat.sum(axis=1).values+multi_coef['const'], name = 'pCPI')
    roll_predict = roll_predict.append(predict)
    print(k)

roll_predict['time']=roll_predict['date']
roll_predict = roll_predict.set_index('time')

fig, ax = plt.subplots(figsize=(8,6))
x= [0,len(roll_predict)]
ax.plot( roll_predict['rCPI'], 'b--', label='rCPI')
ax.plot(roll_predict['pCPI'], 'r--.',label='pCPI')
ax.legend(loc='best',fontsize = 20)
plt.xticks(range(min(x),max(x)+1,3))
plt.grid()
plt.title('CPI_USA roll forecast_WTI1, PPI16, CRB_food6',fontsize = 40)
plt.show()

######## inflection point forecast
def calpct(data_mo,shiftdic,factorsall):
    percentage=[]
    for factor in factorsall:
        calpct =[]
        for i in data_mo.index.values[1:]:
            calpct.append((data_mo[factor].shift(shiftdic[factor])[i]-data_mo[factor].shift(shiftdic[factor])[i-1])/np.abs(data_mo[factor].shift(shiftdic[factor])[i-1]))
        percentage.append(calpct)
    df=pd.DataFrame(percentage).T
    df.columns = factorsall 
    temp=[]
    for i in data_mo.index.values[:-1]:
        temp.append(i+1)
    df.index = np.asarray(temp)
    return df
fac1=['rCPI','pCPI']
df1=roll_predict.copy()
df1=df1.reset_index(drop=True)
shiftdic = dict(zip(fac1,[0]*len(fac1)))
pctroll =calpct(df1,shiftdic,fac1)
pctroll['date']=df1['date'][1:]

#现在在4月,知道2月3月下跌，预测4月涨，4月真的涨.来到了5月，预测5月涨。5月的信号是-1
def signal(pctroll):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1]['rCPI']<0 and pctroll.iloc[i-2]['rCPI']<0:
            if pctroll.iloc[i]['pCPI']>0 and pctroll.iloc[i]['rCPI']>0:
                if pctroll.iloc[i+1]['pCPI']>0:
                    signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['rCPI']>0 and pctroll.iloc[i-2]['rCPI']>0:
            if pctroll.iloc[i]['pCPI']<0 and pctroll.iloc[i]['rCPI']<0:
                if pctroll.iloc[i+1]['pCPI']<0:
                    signal[pctroll.iloc[i+1]['date']]=1
    return signal

def rsignal(pctroll):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1]['rCPI']<0 and pctroll.iloc[i-2]['rCPI']<0:
            if pctroll.iloc[i]['rCPI']>0 and pctroll.iloc[i]['rCPI']>0:
                if pctroll.iloc[i+1]['rCPI']>0:
                    signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['rCPI']>0 and pctroll.iloc[i-2]['rCPI']>0:
            if pctroll.iloc[i]['rCPI']<0 and pctroll.iloc[i]['rCPI']<0:
                if pctroll.iloc[i+1]['rCPI']<0:
                    signal[pctroll.iloc[i+1]['date']]=1
    return signal

#现在在4月,知道2月3月下跌，预测4月涨，4月真的涨-1
def widesignal(pctroll):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1]['rCPI']<0 and pctroll.iloc[i-2]['rCPI']<0:
            if pctroll.iloc[i]['pCPI']>0 and pctroll.iloc[i]['rCPI']>0:
                    signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['rCPI']>0 and pctroll.iloc[i-2]['rCPI']>0:
            if pctroll.iloc[i]['pCPI']<0 and pctroll.iloc[i]['rCPI']<0:
                    signal[pctroll.iloc[i+1]['date']]=1
    return signal
r1=-0.05
r2=0.05
def wblwidesignal(pctroll,r1,r2):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1]['rCPI']<0 and pctroll.iloc[i-2]['rCPI']<0:
            if pctroll.iloc[i]['pCPI']>r1 and pctroll.iloc[i]['rCPI']>0:
                    signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['rCPI']>0 and pctroll.iloc[i-2]['rCPI']>0:
            if pctroll.iloc[i]['pCPI']<r2 and pctroll.iloc[i]['rCPI']<0:
                    signal[pctroll.iloc[i+1]['date']]=1
    return signal
'''
2008-09    1
2010-04   -1
2011-11    1
2012-12    1
2013-03   -1
2013-09    1
2013-12   -1
2016-03    1
2016-05   -1
2017-04    1
2017-08   -1
2017-11    1
'''

r1=-0.05
r2=0.05
def wblwidesignal1(pctroll,r1,r2):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1]['rCPI']<0 and pctroll.iloc[i-2]['rCPI']<0:
            if pctroll.iloc[i]['pCPI']>0 and pctroll.iloc[i]['rCPI']>r1:
                    signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['rCPI']>0 and pctroll.iloc[i-2]['rCPI']>0:
            if pctroll.iloc[i]['pCPI']<0 and pctroll.iloc[i]['rCPI']<r2:
                    signal[pctroll.iloc[i+1]['date']]=1
    return signal

r1=-0.02
r2=0.02

 '''
2008-09    1
2013-09    1
2016-03    1         

wide：
2008-09    1
2011-11    1
2012-12    1
2013-09    1
2016-03    1
2016-05   -1
2017-11    1
'''



############################### Roll Forecast (Freqency:Week) # #######################################
start_date         = '2007-01-01'
end_date           = '2018-08-01'

bestaheaddict1={'WTI': 1.0, 'PPI': 16.0, 'CRB_food': 6.0}
factors =list(bestaheaddict1.keys())
comb               = dicshift_combinations(bestaheaddict1,factors)
data_all           = data3.copy()
ym                 = pd.DataFrame((t.strftime('%Y-%m-%d')[:7] for t in data_all['time']),columns=['ym'])
data_all           = data_all.reset_index(drop=True)
data_all['ym']     = pd.Series(ym['ym'].values,name='ym')
data_all.index     = data_all['ym'].values

i                  = int(np.where(comb.index=='WTI1, PPI16, CRB_food6')[0])

weekdf1=weekdf.copy()
pp2=pp1.copy()
weekdf1['WTI']=weekdf1['WTI'].shift(4)
pp2['PPI']=pp2.shift(16)
weekdf1['CRB_food']=weekdf1['CRB_food'].shift(24)

k1                  = int(np.where(data_all.index=='2008-01')[0])#36
k2                 = int(np.where(data_all.index=='2018-06')[0])#138 201807
roll_predict = pd.DataFrame()

for k in range(k1,k2+1):

    data_mo            = data_all.iloc[:k]
    data_mop           = data_all.iloc[k:][:1]
    d1 =weekdf1[weekdf1['ym']==data_mop.index.values[0]]
    d2 =pp2[pp2['ym']==data_mop.index.values[0]]
    d3 =pd.DataFrame([d2['PPI'].values[0]]*len(d1),columns=['PPI'],index=[d2.index.values[0]]*len(d1))
    d4=pd.concat([d1['WTI'], d3['PPI'],d1[ 'CRB_food']],axis=1)
    
    predict = pd.DataFrame()

    multi_coef = pd.DataFrame(columns = ['coef']) 
    cell = pd.DataFrame() 
    cell['CPI'] = pd.Series(data_mo['CPI'].values, name = 'CPI')
    for factor in factors:
        cell[factor] = data_mo[factor].shift(comb[factor].iloc[i]).values
    cell = cell.dropna(how='any')
    est = sa.OLS(cell['CPI'], sm.add_constant(cell[factors])).fit()
    multi_coef = est.params
    coef1=multi_coef[factors]


    temp_mat        = coef1*d4.iloc[:,0:6]
    predict['pCPI'] = pd.Series(temp_mat.sum(axis=1).values+multi_coef['const'], name = 'pCPI')
    predict.index=d4.index
    predict['date'] = d1['time']
    roll_predict = roll_predict.append(predict)
    print(k)
roll_predict['ym']=roll_predict.index.values
roll_predict.index=roll_predict['date']

date1 =pd.DataFrame(t.strftime('%Y-%m-%d')[:7] for t in roll_predict['date'].values)
data_all['ym']=data_all.index.values
alldf=roll_predict.merge(data_all[['ym','CPI']],on='ym')
alldf.index=alldf['date']

fig, ax = plt.subplots(figsize=(8,6))
#x= [0,len(alldf)]
#plt.plot( data_all['CPI'], 'b--', label='rCPI')
plt.plot(roll_predict['pCPI'].values, 'r--.',label='pCPI')
plt.legend(loc='best',fontsize = 20)
#plt.xticks(range(min(x),max(x)+1,12))
plt.grid()
plt.title('CPI_USA roll forecast_WTI1, PPI16, CRB_food6',fontsize = 40)
plt.show()

plt.plot(roll_predict['pCPI'])
plt.plot(data_all[77:]['CPI'])
plt.show()

fig = plt.figure(figsize=(150,30))
ax1 = fig.add_subplot(111)
ax1.plot(roll_predict['pCPI'])
ax1.set_ylabel('pCPI', fontsize=50)
     
ax2 = ax1.twinx()
ax2.plot(data_all[77:]['CPI'], 'r-')
ax2.set_ylabel('rCPI', color='r', fontsize=50)
plt.legend(fontsize=50)
fig.suptitle('CPI USA', fontsize=50)


######## inflection point forecast
def calpct(data_mo,shiftdic,factorsall):
    percentage=[]
    for factor in factorsall:
        calpct =[]
        for i in data_mo.index.values[1:]:
            calpct.append((data_mo[factor].shift(shiftdic[factor])[i]-data_mo[factor].shift(shiftdic[factor])[i-1])/np.abs(data_mo[factor].shift(shiftdic[factor])[i-1]))
        percentage.append(calpct)
    df=pd.DataFrame(percentage).T
    df.columns = factorsall 
    temp=[]
    for i in data_mo.index.values[:-1]:
        temp.append(i+1)
    df.index = np.asarray(temp)
    return df
fac1=['pCPI']
df1=roll_predict.copy()
df1=df1.reset_index(drop=True)
shiftdic = dict(zip(fac1,[0]*len(fac1)))
pctroll =calpct(df1,shiftdic,fac1)
pctroll['date']=df1['date'][1:]


'''
signal"1": 
    [Peak], Trend changes from going up to going down.
    If both the trend of the recent two week is going up, 
    and we forecast that for the following three week is going down, 
    we forecast the point as a peak and send the signal"1". Vise versa for signal"-1".
signal"-1": 
    [Trough]. Trend changes from going down to going up.
 
'''
def signal(pctroll):
    signal=pd.Series()
    for i in range(3,len(pctroll)):
        if pctroll.iloc[i-1]['pCPI']<0 and pctroll.iloc[i-2]['pCPI']<0  :
            if pctroll.iloc[i]['pCPI']>0 and pctroll.iloc[i+1]['pCPI']>0 and pctroll.iloc[i+2]['pCPI']>0 and pctroll.iloc[i+3]['pCPI']>0:
                signal[pctroll.iloc[i+1]['date']]=-1
        elif pctroll.iloc[i-1]['pCPI']>0 and pctroll.iloc[i-2]['pCPI']>0  :
            if pctroll.iloc[i]['pCPI']<0 and pctroll.iloc[i+1]['pCPI']<0 and pctroll.iloc[i+2]['pCPI']<0 and pctroll.iloc[i+3]['pCPI']<0:
                signal[pctroll.iloc[i+1]['date']]=1
    return signal

print('inflection point signal',signal(pctroll))