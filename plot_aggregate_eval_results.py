# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:14:36 2020

@author: wlwee
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

path_dat = 'C:\\Users\\wlwee\\Documents\\python\\follow_cam_models\\DATA\\evaluationresults_model1_train2/aggregated_eval_results.csv'
path_fig1 = os.path.dirname(path_dat) + '/error_figure_pcutoff.png'
path_fig2 = os.path.dirname(path_dat) + '/error_figure.png'
df = pd.read_csv(path_dat)

df['bodyparts'] = df['bodyparts'].astype('category')

cats = df['bodyparts'].cat.categories

collst1 = ['lightcoral','limegreen','lightsteelblue']
collst2 = ['indianred', 'forestgreen','cornflowerblue']

def make_plot(df, cats, collst1, collst2, path_fig1,path_fig2):
    
    fig, ax = plt.subplots()
    ax.axvline(x=300000, color = 'lightsteelblue')
    coln = 0
    
    for caty in cats:
        ct = caty
        
        dfcat = df[df.bodyparts == ct]
        dfcat = dfcat.sort_values(['Training iterations:'])
        
        iters = dfcat['Training iterations:'].to_list()
        test_error = dfcat[' Test error(px)'].to_list()
        train_error = dfcat[' Train error(px)'].to_list()
        
        test_error_p = dfcat['Test error with p-cutoff'].to_list()
        train_error_p = dfcat['Train error with p-cutoff'].to_list()
        
        ax.plot(iters , train_error_p, color = collst1[coln], label = ct + ' - ' + 'Train error with p-cutoff', linestyle ='--',marker='.')
        ax.plot(iters , test_error_p, color = collst2[coln], label = ct + ' - ' + 'Test error with p-cutoff',marker='.')
        
        ax.legend(loc = 'upper right', frameon=True, bbox_to_anchor=(1.65, 1))
        
        ax.set_ylabel('RMSE ($pixels$)')
        ax.set_xlabel('training iterations')
        plt.autoscale(enable=True, axis=u'both', tight=False)
        
        coln = coln + 1
    
    plt.savefig(path_fig1,dpi=300,bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.axvline(x=300000, color = 'lightsteelblue')
    coln = 0
    
    for caty in cats:
        ct = caty
        
        dfcat = df[df.bodyparts == ct]
        dfcat = dfcat.sort_values(['Training iterations:'])
        
        iters = dfcat['Training iterations:'].to_list()
        test_error = dfcat[' Test error(px)'].to_list()
        train_error = dfcat[' Train error(px)'].to_list()
              
        ax.plot(iters , train_error, color = collst1[coln], label = ct + ' - ' + 'Train error', linestyle ='--',marker='.')
        ax.plot(iters , test_error, color = collst2[coln], label = ct + ' - ' + 'Test error',marker='.')
        
        ax.legend(loc = 'upper right', frameon=True, bbox_to_anchor=(1.65, 1))
        
        ax.set_ylabel('RMSE ($pixels$)')
        ax.set_xlabel('training iterations')
        plt.autoscale(enable=True, axis=u'both', tight=False)
        
        coln = coln + 1
    
    plt.savefig(path_fig2,dpi=300,bbox_inches='tight')
    plt.show()

make_plot(df,cats, collst1, collst2, path_fig1,path_fig2)