# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:09:29 2020

@author: wlwee
"""

import os, glob
import pandas as pd
import deeplabcut
import pathlib
from ruamel.yaml import YAML
import matplotlib.pyplot as plt


def auto_evaluate_model (path_config,
                         num_snapshots,
                         evaluation_dir_to_be_deleted = 'default',
                         first_bodypart = 'default', last_bodypart = 'default'):
    
    '''
    automatically evaluates deeplabcut model snap shots, allows which body parts to be evaluated to be subset by a given start stop range default being all.
    if you wish to re-evaluate the model for a different set of body parts you must delete all the previous evaluation results in the current target evaluation folder
    
    path_config: String
        String containing the full path to the config file for the deeplabcut model
    
    num_snapshots: Integer
        Number of snap shots saved when training a model, set by the 'max_snapshots_to_keep' parameter in deeplabcut.train_network()
    
    evaluation_dir: String
        String containing the full path to the target evaluation directory for the models current Iteration, shuffle, and TrainingFraction
        i.e., 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1'
        if left as default no files will be deleted, if path is given the files in it will be deleted and written over with the new bodyparts.
    
    first_bodypart: Integer
        First body part of range of body parts to evaluate, default is 0
    
    last_bodypart: Integer
        Last body part of a range of body parts to evaluate, default is len(data['bodyparts'])
        
    Example    
    --------
    for evaluating 12 snapshots of a model with body parts 2 to 5
    >>> body_parts_evaluated = auto_evaluate_model(path_config = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/config.yaml',
                                                    num_snapshots = 12,
                                                    evaluation_dir_to_be_deleted = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                                    first_bodypart = 2,
                                                    last_bodypart = 5)
    --------
    
    for evaluating 12 snapshots of a model for all bodyparts
    >>> body_parts_evaluated = auto_evaluate_model(path_config = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/config.yaml',
                                                    num_snapshots = 12,
                                                    evaluation_dir_to_be_deleted = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1')
    --------
    '''
    
    if evaluation_dir_to_be_deleted != 'default':
        if os.path.exists(evaluation_dir_to_be_deleted) == True:
            for f in os.listdir(evaluation_dir_to_be_deleted):
                os.remove(os.path.join(evaluation_dir_to_be_deleted, f))
   
    for i in range(-1 * num_snapshots, 0):
        yaml = YAML()
        mf = pathlib.Path(path_config)
        data = yaml.load(mf)
        
        data['snapshotindex'] = i
        
        yaml.dump(data,mf)
        
        if first_bodypart == 'default':
            first_bodypart = 0
        if last_bodypart == 'default':
            last_bodypart = len(data['bodyparts'])
        
        print(first_bodypart, last_bodypart)
        print('evaluating following bodyparts: ')
        body_parts_evaluated = data['bodyparts'][first_bodypart:last_bodypart]
        print(body_parts_evaluated)
        deeplabcut.evaluate_network(path_config, comparisonbodyparts=data['bodyparts'][first_bodypart:last_bodypart])
    
    return body_parts_evaluated  

def combine_evaluation_results (evaluation_dir,
                                target_dir,
                                bodyparts_descriptor):
    
    '''
    returns the combined data frame of the evaluation results as well as the path to a csv containing it
    
    evaluation_dir: String
        full path to evaluation directory containing model evaluation results for a given Iteration, Shuffle, and TrainingFraction
    
    target_dir: String
        full path to the folder where you want to create a folder that will hold the concatenated results of the evaluated model
    
    bodyparts_descriptor: String
        name descriptor of the body parts evaluated in the evaluation dir
        e.g. 'eyes' or 'mantle' 
    
    Example
    
    --------
    Combining evaluation results
    >>> conc_dfs, conc_eval_csv = combine_evaluation_results ('C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                                                'C:/Users/wlwee/Documents/python/AutomaticOctopus/DATA/auto_evaluate_model',
                                                                'test')
    --------
    
    '''
    
    if os.path.exists(evaluation_dir) == True:
        print('grabbing csvs in: ' + evaluation_dir)
    else:
        print(evaluation_dir + ' DOES NOT EXIST! breaking')
        return '', ''
    
    if os.path.exists(target_dir) == True:
        print('copying csvs to: ' + target_dir + '/' + bodyparts_descriptor)
    else: 
        print(target_dir + ' DOES NOT EXIST! breaking')
        return '', ''
    
    target_dir = target_dir + '/' + bodyparts_descriptor
    if os.path.exists(target_dir) != True:
        os.mkdir(target_dir)
    else:
        print(target_dir + ' already exists!')
        
    csvs = []
    print('found csvs: ')
    for f in os.listdir(evaluation_dir):
        if f.endswith ('.csv'):
            f = os.path.join(evaluation_dir, f)
            csvs.append(f)
            print('   ' + f)
    print('num csvs found is: ' + str(len(csvs)))  
    
    conc_eval_csv = target_dir + '/' + bodyparts_descriptor + '.csv'
    print('creating evaluation csv at: ' + conc_eval_csv)
    df_csvs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df_csvs.append(df)
    conc_dfs = pd.concat(df_csvs)
    conc_dfs = conc_dfs.drop(df.columns[[0]], axis = 1)
    conc_dfs = conc_dfs.sort_values(by = 'Training iterations:')
    
    bodypart_col = [bodyparts_descriptor] * conc_dfs.shape[0]
    conc_dfs['bodypart'] = bodypart_col
    
    conc_dfs.to_csv(conc_eval_csv)
    return conc_dfs, conc_eval_csv

def combine_combined_evaluation_results (lst_path_eval_csvs,
                                         target_dir,
                                         combined_evals_descriptor):
    
    '''
    lst_path_eval_csvs: List
        A list of string containing the full path to csvs containing the combined evaluation results created by the 'combine_evaluation_results()' function
    
    target_dir: String
        A string containing the full path to a directory where the folder containing the combined combined evaluation results will be saved
    
    combined_evals_descriptor: String
        A string containing a descriptor of what the combined evals folder and csv should be called
        i.e., 'eyes_vs_mantle'
        
    Example
    
    --------
    Comparing two seperate bodypart subsets
    >>> combined_conc_dfs, comb_conc_eval_csv = combine_combined_evaluation_results(lst_path_eval_csvs = [conc_eval_csv_2_5, conc_eval_csv_6_all],
                                                                                    target_dir = 'C:/Users/wlwee/Documents/python/AutomaticOctopus/DATA/combined_auto_evaluate_model',
                                                                                    combined_evals_descriptor = 'test')
    --------
    
    '''
    
    print('concatenating the following: ')
    for apath in lst_path_eval_csvs:
        print('   ' + apath)
    df_csvs = []
    for csv in lst_path_eval_csvs:
        df = pd.read_csv(csv)
        df_csvs.append(df)
    conc_dfs = pd.concat(df_csvs)    
    conc_dfs = conc_dfs.drop(df.columns[[0]], axis = 1)
    
    comb_conc_eval_csv = target_dir + '/' + combined_evals_descriptor 
    if os.path.exists(target_dir) == True:
        if os.path.exists(comb_conc_eval_csv) != True:
            print('creating: ' + comb_conc_eval_csv)
            os.mkdir(comb_conc_eval_csv)
        else:
            print(comb_conc_eval_csv + ' already exists!')
    else:
        print(target_dir + ' DOES NOT EXIST! breaking')
        return '', ''
    
    comb_conc_eval_csv = target_dir + '/' + combined_evals_descriptor + '/' + combined_evals_descriptor + '.csv'
    conc_dfs.to_csv(comb_conc_eval_csv)
    
    return conc_dfs, comb_conc_eval_csv

def plot_evaluation_results_of_bodyparts(comb_conc_eval_csv,
                                         use_pcutoff = True,
                                         descriptor = '',
                                         pcutoff = '0.6',
                                         legend_loc = 'upper right',
                                         legend_frame = True,
                                         legend_bbox = (1.5, 1)):
    
    '''
    script to visualize the combined evaluation results for different bodypart subsets csv
    saves a plot into the same directory as the combined bodyparts csv
    uses default matplotlib colors
    
    comb_conc_eval_csv: String
        Full path to the csv containing the combined evaluation results created by 'combine_combined_evaluation_results()' function
    
    use_pcutoff: Boolean
        A boolean descriptor that determines which columns to use when creating the plot, pcutoff or no pcutoff
        
    descriptor: String
        Optional string to add description to the saved name of the plot, useful if you want to save a figure with pcutoff results and no pcutoff results
        
    pcutoff: String
        String of the pcutoff value for your model, default is the deeplabcut default. Only used to alter the title of the plot created
        
    legend_loc: String
        Matplotlib descriptor for where the legend should be placed
    
    legend_frame: Boolean
        Matplotlib descriptor for if the legend should have a box
        
    legend_bbox: tuple
        Tuple that determines the shift of the legend around the legend_loc descriptor
        
    Example
    --------
    plot with pcutoff at default
    >>> plot_evaluation_results_of_bodyparts(comb_conc_eval_csv)
    
    plot without pcutoff
    >>> plot_evaluation_results_of_bodyparts(comb_conc_eval_csv,
                                             use_pcutoff = False,
                                             descriptor = '_no_pcutoff')
    
    --------
    
    '''
    
    df = pd.read_csv(comb_conc_eval_csv)
    df = df.drop(df.columns[[0]], axis = 1)
    df['bodypart'] = df['bodypart'].astype('category')
    bodyparts = df['bodypart'].cat.categories
    
    print('found bodyparts: ')
    for part in bodyparts:
        print('   ' + part)
    
    iterations = df['Training iterations:']
    max_iter = iterations.max()
    print('maximum iteration found: ' + str(max_iter))
        
    fig, ax = plt.subplots()
    
    for part in bodyparts:
        
        dfpart = df[df.bodypart == part]
        dfpart = dfpart.sort_values(['Training iterations:'])
        iters = dfpart['Training iterations:'].to_list()
        
        if use_pcutoff != True:
            test_error = dfpart[' Test error(px)'].to_list()
            train_error = dfpart[' Train error(px)'].to_list()
        else:
            test_error = dfpart['Test error with p-cutoff'].to_list()
            train_error = dfpart['Train error with p-cutoff'].to_list()
        
        test_error_label = part + ' train error'
        train_error_label = part + ' test error'
        
        ax.plot(iters, train_error, label = train_error_label, linestyle ='--',marker='.')
        ax.plot(iters, test_error, label = test_error_label ,marker='.')
        
    if use_pcutoff != True:
        title = 'no p-cutoff'
    else:
        title = 'p-cutoff > ' + str(pcutoff)
    
    plt.title(title)
    ax.set_ylabel('RMSE ($pixels$)')
    ax.set_xlabel('training iterations')
    
    ax.legend(loc = legend_loc, frameon=legend_frame, bbox_to_anchor=legend_bbox)
    plt.autoscale(enable=True, axis=u'both', tight=False)
        
    path_savefig = comb_conc_eval_csv.split('.')[0] + descriptor + '.png'
    print('saving fig at: ' + path_savefig)
    plt.savefig(path_savefig,dpi=300,bbox_inches='tight')
    plt.close()


body_parts_evaluated = auto_evaluate_model(path_config = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/config.yaml',
                                            num_snapshots = 12,
                                            evaluation_dir_to_be_deleted = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                            first_bodypart = 0,
                                            last_bodypart = 2)

conc_dfs, eyes = combine_evaluation_results ('C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                                        'C:/Users/wlwee/Documents/python/AutomaticOctopus/DATA/auto_evaluate_model',
                                                        'eyes')

body_parts_evaluated = auto_evaluate_model(path_config = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/config.yaml',
                                            num_snapshots = 12,
                                            evaluation_dir_to_be_deleted = 'C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                            first_bodypart = 2)

conc_dfs, mantle = combine_evaluation_results ('C:/Users/wlwee/Documents/python/follow_cam_models/MODEL/eyemantle-weert-2020-11-17/evaluation-results/iteration-1/eyemantleNov17-trainset80shuffle1',
                                                            'C:/Users/wlwee/Documents/python/AutomaticOctopus/DATA/auto_evaluate_model',
                                                            'mantle')

combined_conc_dfs, comb_conc_eval_csv = combine_combined_evaluation_results(lst_path_eval_csvs = [eyes, mantle],
                                                                            target_dir = 'C:/Users/wlwee/Documents/python/AutomaticOctopus/DATA/combined_auto_evaluate_model',
                                                                            combined_evals_descriptor = 'eyes_v_mantle')

plot_evaluation_results_of_bodyparts(comb_conc_eval_csv)


