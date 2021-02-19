# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:14:29 2021

@author: wlwee
"""
import os, glob
import shutil

path_vids = glob.glob('harddrive/fhl_flume_singletarget/*/*/*/*.mp4')
target_dir = 'harddrive/fullframe_outliers_extraction'

for vid in path_vids:
    
    new_dir = os.path.join(target_dir, os.path.basename(vid).split('.')[0])
    if os.path.exists(new_dir) != True:
        os.mkdir(new_dir)
        
    new_vid = os.path.join(new_dir, os.path.basename(vid))
    if os.path.exists(new_vid) != True:
        shutil.copyfile(vid, new_vid)