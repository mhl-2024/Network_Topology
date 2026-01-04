# -*- coding: utf-8 -*-

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import sys
import os

from scipy.odr import *
from scipy.stats import linregress
import math 
from shapely.geometry import Point
# %%



def junction_reprocess(df,orderfield):
    records = []

    # Group by ToNode to collect tributaries that flow into the same node
    for junction_node, incoming_group in df.groupby('ToNode'):
        # sort incoming tributaries by order descending
        dftmp = incoming_group.sort_values(by=orderfield, ascending=False)
        
        if dftmp.shape[0] >= 2:
            # keep behavior: take the two largest orders, then make min_max string
            a, b = dftmp[orderfield].iloc[0], dftmp[orderfield].iloc[1]
            id0, id1 = dftmp['NHDPlusID'].iloc[0], dftmp['NHDPlusID'].iloc[1]
            order_min, order_max = (a, b) if a <= b else (b, a)
            records.append({'FromNode': junction_node, 'Tk_order': f"{int(order_min)}_{int(order_max)}",
                            'Strahler1':a,'Strahler2':b,'NHDPlusID1':id0,'NHDPlusID2':id1})
    
    jundf = pd.DataFrame(records, columns=['FromNode', 'Tk_order','Strahler1','Strahler2','NHDPlusID1','NHDPlusID2'])
    return jundf

def Tk_func(jundf,level,outletNHDid): 
    Tk_c_df = []
    rsquared_df = []
    outletnhd = [outletNHDid]
    
    # count occurrences of each Tk_order
    group = jundf.groupby('Tk_order')
    Tk_N = pd.DataFrame(group.size(), columns=['N'])
    Tk_N['Tk_order'] = Tk_N.index
    Tk_N = Tk_N.reset_index(drop=True)
        
    # double counts for same-order pairs "i_i"
    id_pro = []
    for i in range(0, level):
        id0 = str(i) + '_' + str(i)
        id_pro.append(id0)
    for id0 in id_pro:
        Tk_N.loc[Tk_N['Tk_order'] == id0, 'N'] = Tk_N.loc[Tk_N['Tk_order'] == id0, 'N'] * 2
            
    # calculate sum of each record of Tk_N to get N1, N2, N3....
    Tk_N_sum = []
    for j in range(level - 1):
        mask = [True if i[0] == str(j + 1) else False for i in Tk_N['Tk_order']]
        N_sum = Tk_N[mask]['N'].sum()
        Tk_N_sum.append(N_sum)
    # append the highest order
    Tk_N_sum.append(1)

    # add the corresponding Tk_N_sum to df
    for j in range(level):
        Tk_N.loc[(Tk_N['Tk_order'].str[-1] == str(j + 1)), 'Nsum'] = Tk_N_sum[j]

    
    Tk_N['Tk,k+i'] = Tk_N['N'] / Tk_N['Nsum']
        
    # prepare for getting mean Tk,k+i for fitting lines
    Tk_N['order_max'] = Tk_N['Tk_order'].str[-1]
    Tk_N['order_min'] = Tk_N['Tk_order'].str[0]
    Tk_N['order_max'] = Tk_N['order_max'].astype('int')
    Tk_N['order_min'] = Tk_N['order_min'].astype('int')
    Tk_N['order_delta'] = Tk_N['order_max'] - Tk_N['order_min']

    # get mean Tk,k+i for fitting lines
    Tk_mean = []
    num = level - 1
    for i in range(level - 2):
        x1 = Tk_N[Tk_N['order_delta'] == i + 1]['N'].sum()
        x2 = 0
        for j in range(i+1,level):
            x2 = x2 + Tk_N_sum[j]
            
        T = x1/x2
        Tk_mean.append(T)
        num = num - 1
        
    Tk_mean_df = pd.DataFrame({'Tk_mean': Tk_mean})
    
    x = []
    w = []
    for i in range(level - 2):
        x.append(i + 1)
        x_weight = 0
        for j in range(i+1,level):
            x_weight = x_weight + Tk_N_sum[j]
        w.append(np.sqrt(x_weight))

    # log-transform
    Tk_mean_df['Tk_mean_log'] = Tk_mean_df['Tk_mean'].apply(math.log10)
    X = sm.add_constant(x)
    model = sm.WLS(Tk_mean_df['Tk_mean_log'], X, weights=w)
    result = model.fit()                   
               
    Tk_c = np.power(10, result.params[1])
    rsquared = result.rsquared 
                    
    Tk_c_df.append(Tk_c)         
    rsquared_df.append(rsquared)
    
    result_df = pd.DataFrame({'outletNHDid':outletnhd,'Tk_c':Tk_c_df,'rsquared':rsquared_df})
    return result_df

# %%  this section is to calculate Tokunaga parameters 
#===============================   main function ==============================
start = int(sys.argv[1])
end = int(sys.argv[2])
strahler_order=5
orderfield='StreamOrde'
infolder = ''

#========= S1. read the big network =================
df=gpd.read_file(os.path.join(infolder,'NHDhrNetworks.gpkg'))

outletID= df['outletNHDid'].unique()
Tk_df = pd.DataFrame()

for id0 in outletID[start:end]:
    print('now processing basin-----', id0)
    #==== S2. find the network ====
    network = df[df['outletNHDid']==id0]
   
    # === S3. get the junction list of the network ==========
    networkjun=junction_reprocess(network,'StreamOrde')

    # === S4. calculate Tokunaga parameter c ===========
    Tk_df0 = Tk_func(networkjun,strahler_order,id0)
    Tk_df = pd.concat([Tk_df,Tk_df0])
    
# ===== S5. save the files ============
# run on HPC cluster
Tk_df.to_csv('Tk_df_v'+str(start)+'.csv',index=False)
