# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:51:01 2020

@author: wlwee
"""

import deeplabcut
import pathlib
from ruamel.yaml import YAML

path_config = r"C:\Users\wlwee\Documents\python\follow_cam_models\MODEL\eyemantle-weert-2020-11-17\config.yaml"

#deeplabcut.label_frames(path_config)

#deeplabcut.create_training_dataset(path_config)

#deeplabcut.train_network(path_config, displayiters = 100, maxiters = 600000, saveiters = 50000, max_snapshots_to_keep = int(600000/50000))

for i in range(-12,-1):
    yaml = YAML()
    mf = pathlib.Path(path_config)
    data = yaml.load(mf)
    
    data['snapshotindex'] = i
    
    yaml.dump(data,mf)
    
    deeplabcut.evaluate_network(path_config, comparisonbodyparts=data['bodyparts'][2:])
    
    