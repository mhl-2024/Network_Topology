# -*- coding: utf-8 -*-

import pandas as pd
import os


# %%   junction angle aggregation to get basin-averaged values
junfolder = ''
jundf = pd.read_csv(os.path.join(junfolder,'JunctionAngles.csv'))

 # sidebranching angle
sidejun = jundf[jundf['deltaHS']>0]
 # bifurcation angle
bijun = jundf[jundf['deltaHS']==0] 

sideangle = sidejun.groupby('outletNHDid').agg(
    Sidebranch=('angle','mean')).reset_index()
biangle = bijun.groupby('outletNHDid').agg(
    Bifurcation=('angle','mean')).reset_index()
branchangle = jundf.groupby('outletNHDid').agg(
    Branching=('angle','mean')).reset_index()

Network_angles=sideangle.merge(biangle,on='outletNHDid',how='inner')
Network_angles=Network_angles.merge(branchangle,on='outletNHDid',how='inner')

Network_angles.to_csv('JunctionAngles.csv',index=False)


