# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:51:33 2020

@author: wlwee
"""

import deeplabcut
import pandas as pd
import matplotlib.pyplot as plt
import glob, statistics, os, cv2
import numpy as np
#import seaborn as sns

def turn_string_to_float (x):
    x_0 = x.split('_')[0]
    x_1 = x.split('_')[1].split('.')[0]
    x = float(x_0 + '.' + x_1)  
    return x

def turn_to_jpeg (num):
    
    num = str(num)
    num = num.split('.')
    num0 = num[0]
    num1 = num[1]
    num = num0 + '_' + num1 + '.jpeg'
    return num

def turn_string_to_float_cropped_img (x):
    x_0 = x.split('_')[-2]
    x_1 = x.split('_')[-1].split('.')[0]
    x = float(x_0 + '.' + x_1)
    
    return(x)           

def turn_list_of_names_to_jpeg(img_num_list):
    
    img_name_list = []
    for num in img_num_list:
        img_name_list.append(turn_to_jpeg(num))
    return img_name_list

def turn_cropped_image_nums_into_list (crop_num):
    
    nums = crop_num.split('_')
    x0 = float(nums[0])
    y0 = float(nums[1])
    x1 = float(nums[2])
    y1 = float(nums[3])
    
    t1 = nums[4]
    t2 = nums[5]
    time_stamp = float(t1 + '.' + t2)
    
    crop_parameters = [x0, y0, x1, y1, time_stamp]
    
    return crop_parameters
    
def closest(a, K):       
    return min(enumerate(a), key=lambda x: abs(x[1]-K)) 

def get_fps(lst): 
    
    fps_nums = []
    for im in lst:
        fps_nums.append(im)
    fps_calc = []
    try:
        for n in range(0,len(fps_nums),2):
            fps_calc.append(1/(fps_nums[n + 1] - fps_nums[n]))
    except IndexError:
        print('fps_calc weird length')
       
    fps = round(statistics.mean(fps_calc),1)
    
    print('fps: ' + str(fps))
    print('num images: ' + str(len(lst)))
    print('length video: ' + str(len(lst)/fps/60) + ' min')
    '''
    if fps > 10:   
        
        # 7 = fps / n, n = fps / 7
        n = int(round(fps / 7.0, 0))
        print()
        print('fps > 10, subsetting by ' + str(n))
        lst = lst[0:len(lst):n]
        
        fps_nums = []
        for im in lst:
            fps_nums.append(im)
        fps_calc = []
        try:
            for n in range(0,len(fps_nums),2):
                fps_calc.append(1/(fps_nums[n + 1] - fps_nums[n]))
        except IndexError:
            print('fps_calc weird length')
        
        fps = round(statistics.mean(fps_calc),1)
        print('fps: ' + str(fps))
        print('num images: ' + str(len(lst)))
        print('length video: ' + str(len(lst)/fps/60) + ' min')
    '''
    return fps, lst

def condition(x):    
  return x == 0

def sort_dat (dat):
    lst_index = list(dat.index.values.tolist())
    for n in range(0,len(lst_index)):
        lst_index[n] = turn_string_to_float(lst_index[n])
        
    dat.index = lst_index   
    dat = dat.sort_index(axis = 0)
    
    fps, reduced_list_index = get_fps(lst_index)
    
    return dat, fps, lst_index, reduced_list_index

def get_mean_pixel_value(dat):
    
    x_means = []
    y_means = []
    
    for n in range(0, len(dat)):
        
        x_left_eye = dat.iloc[n, 0]
        y_left_eye = dat.iloc[n, 1]
        p_left_eye = dat.iloc[n, 2]
        
        x_right_eye = dat.iloc[n, 3]
        y_right_eye = dat.iloc[n, 4]
        p_right_eye = dat.iloc[n, 5]
        
        x_mantle_tip = dat.iloc[n, 6]
        y_mantle_tip = dat.iloc[n, 7]
        p_mantle_tip = dat.iloc[n, 8]
        
        x_mean_value = 0
        y_mean_value = 0
        m = 0
        
        if p_left_eye > 0.9:
            x_mean_value = x_mean_value + x_left_eye
            y_mean_value = y_mean_value + y_left_eye
            m = m + 1
        if p_right_eye > 0.9:
            x_mean_value = x_mean_value + x_right_eye
            y_mean_value = y_mean_value + y_right_eye
            m = m + 1
        if p_mantle_tip > 0.9:
            x_mean_value = x_mean_value + x_mantle_tip
            y_mean_value = y_mean_value + y_mantle_tip
            m = m + 1
        
        if m > 0:
            x_mean_value = x_mean_value/m
            y_mean_value = y_mean_value/m
        else:
            x_mean_value = 0
            y_mean_value = 0
        
        x_means.append(x_mean_value)
        y_means.append(y_mean_value)
    
    return x_means, y_means

def remove_zeros (x_means, y_means):
    
    x_means_zero_removed = [s for s in x_means if s > 0]
    y_means_zero_removed = [s for s in y_means if s > 0]
    
    return x_means_zero_removed, y_means_zero_removed

def remove_high_velocity_values (x_lst, y_lst, max_inch_per_second, fps):
    
    x_means_high_velocity_removed = x_lst
    y_means_high_velocity_removed = y_lst
    
    index = 1
    # ~20 pixels per inch, fps = 7.0
    # 75 pixels / frame * 1 / 20 inch/pixel * 7 frame / second = 26.25 inches / frame
    # ppf * 1/20 * fps = mips, ppf = mips * 20 / fps
    mips = max_inch_per_second * 20 / fps
    print()
    print ('max pixels per second: ' + str(mips))
    try:
        while index < len(x_means_zero_removed):   
            if abs(x_means_high_velocity_removed[index] - x_means_high_velocity_removed[index - 1]) > mips or abs(y_means_high_velocity_removed[index] - y_means_high_velocity_removed[index - 1]) > mips:
                del x_means_high_velocity_removed[index]
                del y_means_high_velocity_removed[index]
            else:
                index += 1
    except IndexError:
        print('index error')        
    print(index)        
    return x_means_high_velocity_removed, y_means_high_velocity_removed

def crop_images (x_list, y_list, 
                 img_dir_path, img_name_list, crop_dir_path, 
                 crop_x, crop_y, img_dimensions,
                 actually_crop_images):
    
    print()
    print('cropping images from: ' + img_dir_path)
    
    if os.path.exists(crop_dir_path) == False:
        print('making: ' + crop_dir_path)
        os.mkdir(crop_dir_path)
    
    print('cropped images going to: ' + crop_dir_path)
    
    cropped_images_list = []
    for i in range(0, len(x_list)):
        
        img_path = img_dir_path + img_name_list[i]
        crop_save_path = crop_dir_path + img_name_list[i]
        x = int((x_list[i]) - crop_x/2)
        y = int((y_list[i]) - crop_y/2)
        
        img = cv2.imread(img_path)
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0
            
        if x + crop_x < img_dimensions[0]:
            x_c = x + crop_x
        else:
            x_c = img_dimensions[0]
        if y + crop_y < img_dimensions[1]:
            y_c = y + crop_y
        else:
            y_c = img_dimensions[1]        
        
        crop_parameters_name = str(x) + '_' + str(y) + '_' + str(x_c) + '_' + str(y_c) + '_'
        crop_save_path = crop_dir_path + crop_parameters_name + img_name_list[i]
        
        if actually_crop_images == True:
            crop_img = img[y:y_c, x:x_c]
            s_y, s_x, im_c = crop_img.shape
            #print(s_y, s_x)
            
            black_square = np.zeros((crop_x,crop_y,3),np.uint8)
            ax,ay = (s_x - crop_img.shape[1])//2,(s_y - crop_img.shape[0])//2
            #print(ax, ay)
            
            if s_y < crop_y or s_x < crop_x:            
                black_square[ay:crop_img.shape[0]+ay,ax:ax+crop_img.shape[1]] = crop_img
                cv2.imwrite(crop_save_path, black_square)           
            else:
                cv2.imwrite(crop_save_path, crop_img)
        
        cropped_images_list.append(crop_save_path)
    
    return cropped_images_list

def make_many_colored_plot (x_list, y_list, path_background_im, title, plot_save_path, save_plots):
    
    im = plt.imread(path_background_im)
    
    plt.subplots(facecolor='0.7')
    plt.imshow(im)
    
    plt.axis('off')
    
    for i in range(0, len(x_list)-1):
        
        x = [x_list[i], x_list[i+1]]
        y = [y_list[i], y_list[i+1]]
        
        c = [float(i)/float(len(x_list)), #red
                            0.0, # green
                            float(len(x_list) - i)/float(len(x_list)), #blue
                            0.3] #alpha
        
        plt.plot(x,
                 y,
                 color = c)
    
    plt.title(title)
    
    if save_plots == True:
        plt.savefig(plot_save_path)
        
    plt.show()
    
def make_many_colored_plot_w_after (x_list, y_list, x_after_list, y_after_list, path_background_im, title, plot_save_path, save_plots):
    
    im = plt.imread(path_background_im)
    
    plt.subplots(facecolor='0.7')
    plt.imshow(im)
    
    plt.axis('off')
    
    for i in range(0, len(x_list)-1):
        
        x = [x_list[i], x_list[i+1]]
        y = [y_list[i], y_list[i+1]]
        
        c = [float(i)/float(len(x_list)), #red
                            0.0, # green
                            float(len(x_list) - i)/float(len(x_list)), #blue
                            0.5] #alpha
        
        plt.plot(x,
                 y,
                 color = c)
        
    for i in range(0, len(x_after_list)-1):
        
        x = [x_after_list[i], x_after_list[i+1]]
        y = [y_after_list[i], y_after_list[i+1]]

        if float(i)/float(len(x_after_list)) < 0.4:
            c = [float(i)/float(len(x_after_list)), #red
                            float(len(x_after_list) - i)/float(len(x_after_list)), # green
                            0.0, #blue
                            0.5] #alpha
        else:
            c = [0.4, #red
                                0.6, # green
                                0.0, #blue
                                0.5] #alpha
        plt.plot(x,
                 y,
                 color = c)
    
    plt.title(title)
    
    if save_plots == True:
        plt.savefig(plot_save_path)
        
    plt.show()
    
def make_plot(x_list, y_list, path_background_im, title, plot_save_path, save_plots):
    
    im = plt.imread(path_background_im)
    plt.imshow(im)
    plt.plot(x_list, 
             y_list,
             color = (1, 0, 0, 0.3))
    plt.title(title)
    if save_plots == True:
        plt.savefig(plot_save_path)
    plt.show()

def make_video(images, 
                   video_name,
                   fps):
        
    try: 
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, 
                                0, 
                                fps, 
                                (width,height))
        
        print()
        print('making video: ' + video_name)
        
        for image in images:
            img = cv2.imread(image)
            video.write(img)     
        
        cv2.destroyAllWindows()
        video.release()
    
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        video.release()

def get_point_list(dat, n):
    
    p_list = []
    width = dat.shape[1] - 1
    
    for i in range(0,width,3):
        
        x = dat.iloc[n][i]
        y = dat.iloc[n][i+1]
        p = dat.iloc[n][i+2]
        
        xyp = [x, y, p]
    
        if p > 0.9:
            p_list.append(xyp)
            
    return p_list

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def calculate_area (point_list_index):
    
    x = []
    y = []
    
    for point in point_list_index:
        
        if point[2] > 0.9:
            x.append(point[0])
            y.append(point[1])
    
    area = PolyArea(x, y)/(20*20)        
    
    return area

def make_area_plot (pts_list, title, save_plots, plot_save_path):
    
    n = len(pts_list)
    area_list = []
    
    plt.subplots(facecolor='0.7')
    plt.title(title)
    plt.ylabel('poly area (in^2)')
    
    for i in range(0, n - 1):
        area = [calculate_area(pts_list[i]), calculate_area(pts_list[i+1])]
        area_list.append(area[1])
        
        c = [float(i)/float(n), #red
                            0.0, # green
                            float(n - i)/float(n), #blue
                            0.3] #alpha
        
        plt.plot([i, i+1],
                     area,
                     color = c)
    
    if save_plots == True:
        plt.savefig(plot_save_path)
        plt.close()
    else:
        plt.show()
    
    return area_list

def plot_arm_points_img (arm_points, title, path_background_im, plot_save_path, save_plots):
    
    im = plt.imread(path_background_im)
    
    plt.subplots(facecolor='0.7')
    plt.imshow(im)
    plt.title(title)
    plt.axis('off')
    
    nn = 0
    for points in arm_points:
        x = []
        y = []
        
        for i in range(0, len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
            
        c = [float(nn)/float(len(arm_points)), #red
                            0.0, # green
                            float(len(arm_points) - nn)/float(len(arm_points)), #blue
                            0.3] #alpha
            
        plt.fill(x,
                 y,
                 color = c)
        nn = nn + 1
     
    if save_plots == True:
        plt.savefig(plot_save_path)
        plt.close()
    else:
        plt.show()
    
            
food_target = [[1251.5, 583.5],
               [1259, 617.5],
               [1247.5, 809.5],
               [1263.5, 599.5],
               [1238, 336],
               [1261.5, 815.5],
               [1259.5, 591.5],
               [1252, 364],
               [1248, 396],
               [1253.5, 605.5],
               [1259.5, 783.5],
               [1241.5, 785.5],
               [1258, 408],
               [1257, 384],
               [1245.5, 603.5]
               ]

control_target_1 = [[1252, 376],
                    [1256, 420],
                    [1248, 388],
                    [1254, 436],
                    [1245, 599.5],
                    [1260, 382],
                    [1262, 388],
                    [1251.5, 583.5],
                    [1249.5, 589.5],
                    [1246, 418],
                    [1254, 386],
                    [1230, 420],
                    [1259.5, 599.5],
                    [1247.5, 599.5],
                    [1244, 424]
                    ]

control_target_2 = [[1251.5, 805.5],
                    [1259.5, 815.5],
                    [1257.5, 601.5],
                    [1241.5, 791.5],
                    [1249.5, 847.5],
                    [1265.5, 585.5],
                    [1251.5, 821.5],
                    [1253.5, 793.5],
                    [1251.5, 793.5],
                    [1251.5, 783.5],
                    [1263.5, 597.5],
                    [1249.5, 599.5],
                    [1257.5, 797.5],
                    [1256.5, 800.5],
                    [1245.5, 801.5]
                    ]    

starting_frames = [11307.65882192,
                   13594.13063675,
                   19021.85147199,
                   20969.2986958,
                   30343.735758000003,
                   34365.917139050005,
                   11062.658491040002,
                   14803.40860516,
                   16089.706639430002,
                   37756.397373110005,
                   41209.716162820005,
                   122049.5438467,
                   130640.76898944001,
                   12417.61706615,
                   15649.933739130001
                   ]

ending_frames = [11352.78904299,
                 13730.417656380001,
                 19108.280871220002,
                 21005.29904061,
                 30571.023649220002,
                 34421.48909989,
                 11107.801780560001,
                 15920.27644522,
                 18018.01082276,
                 37795.11202963,
                 41258.57377362,
                 123274.55557970001,
                 130780.77033036001,
                 12678.19099047,
                 18084.67134443
                 ]

ending_frames = [s + 10 for s in ending_frames]

eat_yes_no = [True,
              True,
              True,
              True,
              True,
              True,
              True,
              False,
              False,
              True,
              True,
              True,
              True,
              False,
              False]

good_for_tracking_analysis = [True,
                              True,
                              True,
                              False,
                              True,
                              True,
                              True,
                              False,
                              False,
                              True,
                              True,
                              False,
                              True,
                              False,
                              False]

starting_frame_names = [turn_to_jpeg(s) for s in starting_frames]

title_names =  ['9/3 trial 01',
                '9/3 trial 02',
                '9/4 trial 03',
                '9/4 trial 04',
                '9/5 trial 05',
                '9/5 trial 06',
                '9/6 trial 07',
                '9/6 trial 08',
                '9/6 trial 09',
                '9/7 trial 10', 
                '9/7 trial 11',
                '9/8 trial 12',
                '9/8 trial 13',
                '9/9 trial 14',
                '9/9 trial 15'
                 ]

save_plots = True

h5_extension = '*.h5'
path_imgs_root_dir = 'D:/FHL_Flume/slaanesh_plume_tracking/'
sub_name = '_plume_tracking/'

save_plot_path = 'C:/Users/wlwee/Documents/python/fhl_flume_dlc/PLOTS/slaanesh_trials/9_15_slaanesh_trial_plots/'

path_config = r'C:\Users\wlwee\Documents\python\fhl_flume_dlc\MODELS\arms-weertman-2020-09-14\config.yaml'

path_img_dirs = [path_imgs_root_dir + '9_3' + sub_name + 'slaanesh_9_3_tracking_01/',
                 path_imgs_root_dir + '9_3' + sub_name + 'slaanesh_9_3_tracking_02/',
                 path_imgs_root_dir + '9_4' + sub_name + 'slaanesh_9_4_tracking_01/',
                 path_imgs_root_dir + '9_4' + sub_name + 'slaanesh_9_4_tracking_02/',
                 path_imgs_root_dir + '9_5' + sub_name + 'slaanesh_9_5_tracking_01/',
                 path_imgs_root_dir + '9_5' + sub_name + 'slaanesh_9_5_tracking_02/',
                 path_imgs_root_dir + '9_6' + sub_name + '9_6_tracking_01/',
                 path_imgs_root_dir + '9_6' + sub_name + '9_6_tracking_02/',
                 path_imgs_root_dir + '9_6' + sub_name + '9_6_tracking_03/',
                 path_imgs_root_dir + '9_7' + sub_name + '9_7_tracking_01/',
                 path_imgs_root_dir + '9_7' + sub_name + '9_7_tracking_02/',
                 path_imgs_root_dir + '9_8' + sub_name + '9_8_tracking_01/',
                 path_imgs_root_dir + '9_8' + sub_name + '9_8_tracking_02/',
                 path_imgs_root_dir + '9_9' + sub_name + '9_9_tracking_01/',
                 path_imgs_root_dir + '9_9' + sub_name + '9_9_tracking_02/'
                 ]

extract_images = True
actually_crop_images = False
make_a_video = False
analyze_video_w_dlc = True

im_dimension = [1920, 1200]
crop_x = 500
crop_y = 500
extracted_img_root_dir = 'C:/Users/wlwee/Documents/python/fhl_flume_dlc/DATA/slaanesh_tracking_extracted/'
extracted_img_dirs = []
for name in path_img_dirs:
    
    name = name.split('/')[-2] + '/'
    name = extracted_img_root_dir + name
    extracted_img_dirs.append(name)
    
max_inch_per_second = 24

trial_dict = []
for ps, ns, strts, eds, ets, gft in zip (path_img_dirs, 
                                    title_names,
                                    starting_frames,
                                    ending_frames,
                                    eat_yes_no,
                                    good_for_tracking_analysis):
    
    trial = {'path trial': ps,
             'name trial': ns,
             'start': strts,
             'end': eds,
             'time trial': eds - strts,
             'eat T or F': ets,
             'good data': gft}
    trial_dict.append(trial)
    

n = 0
for a_dir in path_img_dirs:
    
    targets_list = [food_target[n],
                    control_target_1[n],]
    
    trial = trial_dict[n]
        
    path_background_im = a_dir + starting_frame_names[n]
    a_dir_h5 = a_dir + h5_extension
    path_h5 = glob.glob(a_dir_h5)[0]  
    title = os.path.basename(path_h5)
    
    # calculate the time of the trial we care about
    total_time_trial = round((ending_frames[n] - starting_frames[n])/60,1)
    
    # read the data frame
    dat = pd.read_hdf(path_h5)
    dat, fps, index_list, r_index_list = sort_dat(dat)
    
    # get the frame where the data frame is to be cut up around
    drop_after_index, check_time = closest(r_index_list, starting_frames[n])
    keep_until_index, check_time = closest(r_index_list, ending_frames[n])
    
    # chop up the data frame 
    dat_after = dat.iloc[keep_until_index:]
    dat = dat.iloc[drop_after_index:keep_until_index]
    
    # get lists that contain the data frame index reorganized to contain the actual image name
    img_after_name_list = turn_list_of_names_to_jpeg(dat_after.index)
    img_name_list = turn_list_of_names_to_jpeg(dat.index)
    
    # get the mean pixel values of the left eye, right eye, mantle tip from DLC
    x_means_after, y_means_after = get_mean_pixel_value(dat_after)
    x_means, y_means = get_mean_pixel_value(dat)
    
    #remove all frames where there is a zero value, where the position of the octopus is uncertain
    x_means_zero_removed_after, y_means_zero_removed_after = remove_zeros(x_means_after, y_means_after)
    x_means_zero_removed, y_means_zero_removed = remove_zeros(x_means, y_means)
    
    # remove all frames where the position of the octopus unrealistically jumps
    x_m_a, y_m_a = remove_high_velocity_values(x_means_zero_removed_after,
                                                y_means_zero_removed_after,
                                                max_inch_per_second,
                                                fps)
    x_m, y_m = remove_high_velocity_values(x_means_zero_removed,
                                            y_means_zero_removed,
                                            max_inch_per_second,
                                            fps)
    
    # get conditional statement on data quality
    if trial['eat T or F'] == True:
        eat = 'ate'     
    else:
        eat = 'no eat'
        
    if trial['good data'] == True:
        data = 'good'            
    else:
        data = 'bad'
        
    title = title_names[n] + ' - ' + str(total_time_trial) + ' min - ' + eat + ' - ' + data
    
    '''
    make_plot(x_means_zero_removed, 
              y_means_zero_removed, 
              path_background_im)
    '''
    
    plot_path = save_plot_path + title_names[n].split('/')[0] + '_' + title_names[n].split('/')[1] +  '.png'
    
    '''
    make_plot(x_m, 
              y_m,
              path_background_im,
              title,
              plot_path,
              save_plots)
    '''
    if trial['good data'] == True:
        '''
        make_many_colored_plot_w_after(x_m, 
                                        y_m,
                                        x_m_a,
                                        y_m_a,
                                        path_background_im,
                                        title,
                                        plot_path,
                                        save_plots)
        '''
        
        if extract_images == True:
            
            cropped_images_list = crop_images(x_list = x_m, 
                                                y_list = y_m, 
                                                img_dir_path = a_dir, 
                                                img_name_list = img_name_list, 
                                                crop_dir_path = extracted_img_dirs[n], 
                                                crop_x = crop_x, 
                                                crop_y = crop_y, 
                                                img_dimensions = im_dimension,
                                                actually_crop_images = actually_crop_images)
            
            cropped_images_nums = [os.path.basename(s).split('.')[0] for s in cropped_images_list]
            
            cropped_img_parameters = []
            for ig in cropped_images_nums:
                cropped_img_parameters.append(turn_cropped_image_nums_into_list(ig))
            
            vid_base_name = extracted_img_dirs[n].split('/')[-2] + '.avi'
            vid_dir = extracted_img_dirs[n] + extracted_img_dirs[n].split('/')[-2] + '_video/'
            
            if os.path.exists(vid_dir) == False:
                os.mkdir(vid_dir)
            
            vid_name = vid_dir + vid_base_name
            
            if make_a_video == True:
                make_video(images = cropped_images_list,
                           video_name = vid_name,
                           fps = fps)
                
            if analyze_video_w_dlc == True:
                
                video_data_folder = vid_dir + 'dlc_analysis/'
                if os.path.exists(video_data_folder) == False:
                    os.mkdir(video_data_folder)
                
                deeplabcut.analyze_videos(path_config,
                                          [vid_name],
                                          destfolder = video_data_folder,
                                          videotype = '.avi')
                deeplabcut.create_labeled_video(path_config,
                                                [vid_name],
                                                draw_skeleton=True,
                                                destfolder = video_data_folder)
                deeplabcut.plot_trajectories(path_config,
                                             [vid_name],
                                             destfolder= video_data_folder)
                
                arm_h5 = glob.glob(video_data_folder + '*.h5')[0]
                dat_arm_h5 = pd.read_hdf(arm_h5)
                
                length_dat_h5 = dat_arm_h5.shape[0] - 1
    
                name = os.path.basename(arm_h5).split('slaanesh')
                if len(name) > 1:
                    name = name[1]
                else:
                    name = name[0]
                name = name.split('DLC')[0]
                
                points_list = []
                for i in range(0, length_dat_h5):
                    points = get_point_list(dat_arm_h5, i)
                    points_list.append(points)
                
                
                plot_save_path = video_data_folder + name
                
                make_many_colored_plot_w_after(x_m, 
                                        y_m,
                                        x_m_a,
                                        y_m_a,
                                        path_background_im,
                                        title,
                                        plot_save_path + '_model_1',
                                        save_plots)
                
                area_list = make_area_plot(points_list,
                                           title + '- area arms', 
                                           save_plots, 
                                           plot_save_path + '_model_2')
                
                corrected_arm_points = []
                for i in range(0, len(points_list)):
                    xy_arm = []
                    for m in range(0, len(points_list[i])):
                        
                        xy_arm_hold = []
                        if points_list[i][m][2] > 0.9:
                            x_cor = points_list[i][m][0] + cropped_img_parameters[i][0]
                            y_cor = points_list[i][m][1] + cropped_img_parameters[i][1]
                            
                            xy_arm_hold.append(x_cor)
                            xy_arm_hold.append(y_cor)
                        xy_arm.append(xy_arm_hold)
                    corrected_arm_points.append(xy_arm)
                    
                plot_arm_points_img(corrected_arm_points, 
                                    title + '- arms', 
                                    path_background_im,
                                    plot_save_path + '_model_2_on_im',
                                    save_plots)
                    
                
    n = n + 1

plt.close()
    