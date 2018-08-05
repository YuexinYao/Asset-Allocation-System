# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:35:35 2018

@author: Yuexin Yao
"""
'''
Generate inflection points' signals for China's GDP YoY
'''
from WindPy import *
w.start()
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression 
import scipy.stats as stats
import numpy as np
import matplotlib 
%matplotlib qt5

###### Function Defination
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


def signaluud(pctroll,fac):
    signal=pd.Series()
    for i in range(2,len(pctroll)):
        if pctroll.iloc[i-1][fac]<0 and pctroll.iloc[i-2][fac]<0:
            if pctroll.iloc[i][fac]>0:
                signal[pctroll.iloc[i]['date']]=-1#i-2: down; i-1:down ; i:upsignal-1
        elif pctroll.iloc[i-1][fac]>0 and pctroll.iloc[i-2][fac]>0:
            if pctroll.iloc[i][fac]<0:
                signal[pctroll.iloc[i]['date']]=1#i-2: up; i-1:up ; i:down signal+1
    return signal

def signalud(pctroll,fac):
    signal=pd.Series()
    for i in range(1,len(pctroll)):
        if pctroll.iloc[i-1][fac]<0:
            if pctroll.iloc[i][fac]>0 :
                signal[pctroll.iloc[i]['date']]=-1 #i-1: down; i:up signal -1
        elif pctroll.iloc[i-1][fac]>0:
            if pctroll.iloc[i][fac]<0 :
                signal[pctroll.iloc[i]['date']]=1 #i-1:up i:down signal -+1
    return signal

def signaluudd(pctroll,fac):
    signal=pd.Series()
    for i in range(3,len(pctroll)):
        if pctroll.iloc[i-1][fac]>0 and pctroll.iloc[i-2][fac]<0 and pctroll.iloc[i-3][fac]<0:
            if pctroll.iloc[i][fac]>0 :
                signal[pctroll.iloc[i+1]['date']]=-1 #i-3:down; i-2: down; i-1:up ; i:up signal-1
        elif pctroll.iloc[i-1][fac]<0 and pctroll.iloc[i-2][fac]>0 and pctroll.iloc[i-3][fac]>0:
            if pctroll.iloc[i][fac]<0 :
                signal[pctroll.iloc[i]['date']]=1#i-3:up; i-2: up; i-1:down; i:down signal+1
    return signal

def dicshift_combinations(bestaheaddict,factors):
    cbn0 = []
    for factor in factors:
        t    = int(bestaheaddict[factor])
        temp = [t]
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

def 有常项多因子回归(fac,data_mo,yname,bestaheaddict):   
    combination = dicshift_combinations(bestaheaddict,fac)
    result = []
    for i in range(len(combination)):
        reg = pd.DataFrame(columns = ['coef', 't', 'p']) 
        cell = pd.DataFrame() 
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
##### Industrial added value YoY
dataindustry                = w.edb("M0000545", "2001-01-01", "2018-06-30")
industry1   = pd.DataFrame(dataindustry.Data).T
industry1=industry1.rename(columns={0:'industry'})
industry1['time'] = pd.to_datetime(dataindustry.Times)
industry1['industrymid']=industry1['industry'].rolling(6).mean()
ym                = pd.DataFrame(t.strftime('%Y-%m-%d')[:7] for t in industry1['time'])
industry1['ym'] = ym
industry2 = industry1[['industrymid','ym']]

jidu      =w.edb("M0039354", "2001-01-01", "2018-06-30")
jidu1   = pd.DataFrame(jidu.Data).T
jidu1['time'] = pd.to_datetime(jidu.Times)
ym1                = pd.DataFrame(t.strftime('%Y-%m-%d')[:7] for t in jidu1['time'])
jidu1['ym'] = ym1
jidu1=jidu1.rename(columns={0:'GDP'})
dfall =jidu1.merge(industry1,on='ym')

fig = plt.figure(figsize=(150,30))
plt.plot(dfall['GDP'],color='r')   
plt.plot(dfall['industrymid'],label='industrymid') 
plt.legend()
plt.show()

for k in range(1,24):
    industry1['industrymid']=industry1['industry'].rolling(k).mean()
    dfall =jidu1.merge(industry1,on='ym')
    lag=[]
    cor=[]
    for i in range(24):
        lag.append(i)
        cor.append(dfall['GDP'].corr(dfall['industrymid'].shift(i)))
    plt.figure(1,figsize=(10,5))
    plt.plot(lag,cor)
    print('k=',k,'最佳组合',lag[ cor.index(max(cor))],max(cor)) #industry shift0 rolling6 


fac1=['industrymid']
fac=fac1[0]
df1=industry1.copy().dropna()
df1=df1.reset_index(drop=True)
shiftdic = dict(zip(fac1,[0]*len(fac1)))
pctrollid =calpct(df1,shiftdic,fac1)
pctrollid['date']=df1['time'][1:]

signalud_id = signalud(pctrollid,fac1[0]) #level1
signaluud_id = signaluud(pctrollid,fac1[0])*2#level2
signaluudd_id = signaluudd(pctrollid,fac1[0])*3#level3
allsignal_id = pd.concat([signalud_id,signaluud_id,signaluudd_id],axis=1)
allsignal_id.columns=['ud_id','uud_id','uudd_id']

##### 工业增加值
datapmi                = w.edb("M0017126", "2001-01-01", "2018-06-30")
pmi1   = pd.DataFrame(datapmi.Data).T
pmi1['time'] = pd.to_datetime(datapmi.Times)
pmi1=pmi1.rename(columns={0:'pmi'})
pmi1['pmimid']=pmi1['pmi'].rolling(6).mean()
ympmi                = pd.DataFrame(t.strftime('%Y-%m-%d')[:7] for t in pmi1['time'])
pmi1['ym'] = ympmi

dfpmi = jidu1.merge(pmi1,on='ym')

fig = plt.figure(figsize=(150,30))
plt.plot(dfpmi['GDP'],color='r')   
plt.plot(dfpmi['pmimid'],label='pmimid') 
plt.legend()
plt.show()

for k in range(1,24):
    pmi1['pmimid']=pmi1['pmi'].rolling(k).mean()
    df0 =jidu1.merge(pmi1,on='ym')
    lag=[]
    cor=[]
    for i in range(24):
        lag.append(i)
        cor.append(df0['GDP'].corr(df0['pmimid'].shift(i)))
    plt.figure(1,figsize=(10,5))
    plt.plot(lag,cor)
    print('k=',k,'最佳组合',lag[ cor.index(max(cor))],max(cor)) #pmi shift0 rolling/9 0.89/6 0.84

fac2=['pmimid']
df2=pmi1.copy().dropna()
df2=df2.reset_index(drop=True)
shiftdic = dict(zip(fac2,[0]*len(fac2)))
pctrollpmi =calpct(df2,shiftdic,fac2)
pctrollpmi['date']=df2['time'][1:]

signalud_pmi = signalud(pctrollpmi,fac2[0])#level1
signaluud_pmi = signaluud(pctrollpmi,fac2[0])*2#level2
signaluudd_pmi = signaluudd(pctrollpmi,fac2[0])*3#level3
allsignal_pmi=pd.concat([signalud_pmi,signaluud_pmi,signaluudd_pmi],axis=1)
allsignal_pmi.columns=['ud_pmi','uud_pmi','uudd_pmi']
signalud_pmi_and_id = pd.concat([signalud_id,signalud_pmi],axis=1).dropna()
signal_pmi_and_id = pd.concat([allsignal_id,allsignal_pmi],axis=1)

#Regeression : Seasonly

data_all = dfpmi.merge(industry1,on='ym')[['GDP','ym','pmimid','industrymid']]
bestaheaddict={'GDP':0,'pmimid':0,'industrymid':0}
fac=['pmimid','industrymid']
yname='GDP'

data_all.index = data_all['ym']
有常项多因子回归(fac,data_all,yname,bestaheaddict)[0]['coef']
k1                  = int(np.where(data_all.index=='2006-03')[0])
k2                 = int(np.where(data_all.index=='2018-06')[0])
roll_predict = pd.DataFrame()
for k in range(k1,k2+1):
    data_mo            = data_all.iloc[:k]
    data_mop           = data_all.iloc[k:][:1]#下一个月
    
    predict = pd.DataFrame()
    multi_coef      = 有常项多因子回归(fac,data_all,yname,bestaheaddict)[0]['coef']
    predict['date'] = data_mop.index.values
    predict['rGDP'] =pd.Series(data_mop['GDP'].values, name = 'rGDP')
    temp_mat        = multi_coef[fac].T*data_mop[fac]
    predict['pGDP'] = pd.Series(temp_mat.sum(axis=1).values+ multi_coef['const'], name = 'pGDP')
    roll_predict = roll_predict.append(predict)
    print(k)

roll_predict['time']=roll_predict['date']
roll_predict = roll_predict.set_index('time')
fig, ax = plt.subplots(figsize=(8,6))
x= [0,len(roll_predict)]
ax.plot( roll_predict['rGDP'], 'b--', label='rGDP')
ax.plot(roll_predict['pGDP'], 'r--.',label='pGDP')
ax.legend(loc='best',fontsize = 20)
plt.xticks(range(min(x),max(x)+1,2))
plt.grid()
plt.title('PMI 工业增加值 6个月移动平均 预测 GDP forecast',fontsize = 30)

fac3=['pGDP']
df3=roll_predict.copy().dropna()
df3=df3.reset_index(drop=True)
shiftdic3 = dict(zip(fac3,[0]*len(fac3)))
pctrollgdp =calpct(df3,shiftdic3,fac3)
pctrollgdp['date']=df3['date'][1:]

signalud_pgdp = signalud(pctrollgdp,fac3[0])#level1
signaluud_pgdp = signaluud(pctrollgdp,fac3[0])*2#level2
signaluudd_pgdp = signaluudd(pctrollgdp,fac3[0])*3#level3
allsignal = pd.concat([signalud_pgdp,signaluud_pgdp,signaluudd_pgdp],axis=1)
allsignal.columns=['ud','uud','uudd']


#Regression:by using high freqency (monthly data)
gp =pmi1.merge(industry1,on='ym')[['ym','time_x','pmimid','industrymid']]
data_all.index = data_all['ym']
有常项多因子回归(fac,data_all,yname,bestaheaddict)[0]['coef']
k1                  = int(np.where(data_all.index=='2006-03')[0])
k2                 = int(np.where(data_all.index=='2018-06')[0])
roll_predict1 = pd.DataFrame()
for k in range(k1,k2+1):
    data_mo            = data_all.iloc[:k1]
    data_mop           = data_all.iloc[k1:][:1]#next period
    data_predict= gp[3*k:3*k+3]
    
    predict = pd.DataFrame()
    multi_coef      = 有常项多因子回归(fac,data_all,yname,bestaheaddict)[0]['coef']
    predict['date'] = data_predict['ym']
    predict.index=predict['date']
    temp_mat        = multi_coef[fac].T*data_predict[fac]
    predict['pGDP'] = pd.DataFrame(temp_mat.sum(axis=1).values+ multi_coef['const'],index=predict['date'])
    roll_predict1 = roll_predict1.append(predict)
    print(k)
roll_predict1

jidu2 =jidu1.copy()
jidu2.index=jidu2['ym']
jidu2[20:]


fig = plt.figure(figsize=(150,30))
plt.plot(roll_predict1['pGDP'],label='预测GDP')
plt.plot(jidu2['GDP'][20:],label='GDP')
x= [0,len(roll_predict1)]
plt.xticks(range(min(x),max(x)+1,6))
plt.title('高频预测 GDP',fontsize=30)
plt.legend()
plt.grid()
plt.show()  

fac3=['pGDP']
df4=roll_predict1.copy().dropna()
df4=df4.reset_index(drop=True)
shiftdic3 = dict(zip(fac3,[0]*len(fac3)))
pctrollgdp =calpct(df4,shiftdic3,fac3)
pctrollgdp['date']=df4['date'][1:]

signalud_pgdp1 = signalud(pctrollgdp,fac3[0])#跌然后涨
signaluud_pgdp1 = signaluud(pctrollgdp,fac3[0])*2#2连跌然后涨
signaluudd_pgdp1 = signaluudd(pctrollgdp,fac3[0])*3#两连跌然后涨，然后涨
allsignal1 = pd.concat([signalud_pgdp1,signaluud_pgdp1,signaluudd_pgdp1],axis=1)
allsignal1.columns=['ud','uud','uudd']   
    
