# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:53:21 2020

@author: wlwee
"""
import deeplabcut
import glob, os, cv2, statistics, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import statistics as st

def from_crop_parameters_get_x1y1 (s):
    x1 = float(s.split('_')[0])
    y1 = float(s.split('_')[1])
    
    return x1, y1

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    
    to rotate clockwise give angle as a negative
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

def between_points (x1,x2,t):
    x = x1 + (x2-x1) * t
    return x

def interpolate (x1,x2,y1,y2,t):
    x = between_points(x1,x2,t)
    if x1 == x2:
        y = y1
    else:
        y = y1 + (x - x1)*((y2-y1)/(x2-x1))
        
    return x, y

def Left_index(points): 
      
    ''' 
    Finding the left most point 
    '''
    minn = 0
    for i in range(1,len(points)): 
        if points[i][0] < points[minn][0]: 
            minn = i 
        elif points[i][0] == points[minn][0]: 
            if points[i][1] > points[minn][1]: 
                minn = i 
    return minn 

def orientation(p, q, r): 
    ''' 
    To find orientation of ordered triplet (p, q, r).  
    The function returns following values  
    0 --> p, q and r are colinear  
    1 --> Clockwise  
    2 --> Counterclockwise  
    '''
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]) 
  
    if val == 0: 
        return 0
    elif val > 0: 
        return 1
    else: 
        return 2
    
def convexHull(points, n): 
      
    # There must be at least 3 points  
    if n < 3: 
        return
  
    # Find the leftmost point 
    l = Left_index(points) 
  
    hull = [] 
      
    ''' 
    Start from leftmost point, keep moving counterclockwise  
    until reach the start point again. This loop runs O(h)  
    times where h is number of points in result or output.  
    '''
    p = l 
    q = 0
    while(True): 
          
        # Add current point to result  
        hull.append(p) 
  
        ''' 
        Search for a point 'q' such that orientation(p, x,  
        q) is counterclockwise for all points 'x'. The idea  
        is to keep track of last visited most counterclock-  
        wise point in q. If any point 'i' is more counterclock-  
        wise than q, then update q.  
        '''
        q = (p + 1) % n 
  
        for i in range(n): 
              
            # If i is more counterclockwise  
            # than current q, then update q  
            if(orientation(points[p],  
                           points[i], points[q]) == 2): 
                q = i 
  
        ''' 
        Now q is the most counterclockwise with respect to p  
        Set p as q for next iteration, so that q is added to  
        result 'hull'  
        '''
        p = q 
  
        # While we don't come to first point 
        if(p == l): 
            break
  
    # Print Result
    hull_list = []
    for each in hull: 
        hull_list.append([points[each][0], points[each][1]])
    
    return hull_list

def get_fps(lst): 
    
    print()
    print('getting fps')
    
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
    
    print()
    print('fps: ' + str(fps))
    print('num images: ' + str(len(lst)))
    print('length video: ' + str(len(lst)/fps/60) + ' min')
    print()
    
    return fps

def make_many_colored_plot (x_list, y_list, path_background_im, title, plot_save_path, save_plots):
    
    print()
    print('making trajectory plot')
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
    
    print('plot made')
    print() 
    
    plt.axis('off')
    plt.title(title)
    
    if save_plots == True:
        plt.savefig(plot_save_path)
        
    plt.close()
        

def get_mean_pixel_value(dat):
    
    x_means = []
    y_means = []
    
    print('getting mean pixel values from dlc predictions')
    for n in range(0, len(dat)):
        
        x_left_eye = float(dat.iloc[n, 1])
        y_left_eye = float(dat.iloc[n, 2])
        p_left_eye = float(dat.iloc[n, 3])
        
        x_right_eye = float(dat.iloc[n, 4])
        y_right_eye = float(dat.iloc[n, 5])
        p_right_eye = float(dat.iloc[n, 6])
        '''
        x_mantle_tip = float(dat.iloc[n, 7])
        y_mantle_tip = float(dat.iloc[n, 8])
        p_mantle_tip = float(dat.iloc[n, 9])'''
        
        x_mean_value = 0
        y_mean_value = 0
        m = 0
        
        if p_left_eye > 0.6:
            x_mean_value = x_mean_value + x_left_eye
            y_mean_value = y_mean_value + y_left_eye
            m = m + 1
        if p_right_eye > 0.6:
            x_mean_value = x_mean_value + x_right_eye
            y_mean_value = y_mean_value + y_right_eye
            m = m + 1
            '''
        if p_mantle_tip > 0.6:
            x_mean_value = x_mean_value + x_mantle_tip
            y_mean_value = y_mean_value + y_mantle_tip
            m = m + 1'''
        
        if m > 0:
            x_mean_value = x_mean_value/m
            y_mean_value = y_mean_value/m
        else:
            x_mean_value = 0
            y_mean_value = 0
        
        x_means.append(x_mean_value)
        y_means.append(y_mean_value)
    
    print('number of x and y values: ' + str(len(x_means)) + ', ' + str(len(y_means)))
    return x_means, y_means

def recombine_frame_str_to_float (frame_str):
    
    f = frame_str.split('_')[0] + '.' + frame_str.split('_')[1]
    f = float(f)
    
    return(f)

def closest(lst, K):  
    
    findex = min(range(len(lst)), key = lambda i: abs(lst[i] - K))
    fvalue = lst[findex]
    
    return findex, fvalue

#to set arm touch or mouth
#t = 1 for mouth, t = 0 for arm
def iterate_touch_csv (touch_dat,t):
    
    if t > 1:
        print('t needs to be a 0 or a 1 (arm touch, mouth touch)')
    
    for i in touch_dat.itertuples():
        
        if i[2] == 1 and i[-1] == t:
            
            return recombine_frame_str_to_float(i[1])

def crop_img (img_path, x_value, y_value, crop_size):
    
    img = cv2.imread(img_path)
    
    img_dimensions = img.shape
    
    #x,y value minus half of the crop size gives upper right
    x = int((x_value) - crop_size/2)
    y = int((y_value) - crop_size/2)
    
    xr, yr = x,y
    
    #checks if x + crop_size falls outside of the image being cropped
    if x + crop_size < img_dimensions[1]:
        #x + crop falls within the image, no problem x_c = x + crop_size
        x_c = x + crop_size
    else:
        #x + crop falls outside the image so we need to limit x_c at the edge of the image
        x_c = img_dimensions[1]
        
    #same logic as above
    if y + crop_size < img_dimensions[0]:
        y_c = y + crop_size
    else:
        y_c = img_dimensions[0] 

    #if x < 0 that means x falls outside of the image on the negative side, so we set to zero
    if x < 0:
        x = 0
    #same logic as above
    if y < 0:
        y = 0

    #saves (x,y)(x_c,y_c) as a string for later use
    crop_parameters_name = str(x) + '_' + str(y) + '_' + str(x_c) + '_' + str(y_c)
    
    #crop the image out of the big image
    #starts at (x,y) goes to (x_c,y_c)
    crop_img = img[y:y_c, x:x_c]
    
    #makes a black background
    black_square = np.zeros((crop_size,crop_size,3),np.uint8)
    
    #puts the cropped image on top of the black background keeping center point of octopus eyes centered
    #x - xr gives the distances from the top left corner of the black background the paste is placed in
    #x_c - xr gives the distance from the bottom left corner of the black background the paste is placed in
    black_square[int(y - yr):int(y_c - yr),
                 int(x - xr):int(x_c - xr)] = crop_img
    
    return black_square, crop_parameters_name
    
def make_cropped_video(mean_values_tuple,
                       frame_path_list,
                       crop_size,
                       video_name,
                       fps):
    
    try:     
        
        crop_parameters_list = []
        
        video = cv2.VideoWriter(video_name, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (crop_size, crop_size))
    
        print()
        print('making video: ' + video_name)
        
        n = 0
        for image_path in frame_path_list:
            
            img, crop_parameters = crop_img(image_path,
                                            x_value = mean_values_tuple[0][n],
                                            y_value = mean_values_tuple[1][n],
                                            crop_size = crop_size)
            
            video.write(img)   
            
            crop_info = str(crop_parameters)
            
            crop_parameters_list.append(crop_info)
            n = n + 1
    
        cv2.destroyAllWindows()
        video.release()
        print('video made')
        print()
        
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        video.release()
        
    return crop_parameters_list

def analyze_and_clean_arm_dat(dat_mantle_dlc, adir, pcutoff, body_part_line_list, play_back_speed):
    
    #gets a path to the raw video that the previous script created when making the above csv
    vid = glob.glob(adir + '\\*.mp4')[0]
    
    #gets the crop parameters from mantle dlc as a string list
    print('getting crop parameters')
    crop_parameters = dat_mantle_dlc['crop_parameters']
    #gets the (x1,y1) crop parameters as a list of 1x2 floats
    crop_x1y1_parameters = [from_crop_parameters_get_x1y1(s) for s in crop_parameters]
    #gets a list of the paths to the original images cropped from
    #img_paths_list = dat_mantle_dlc['img_paths'].to_list()
    
    #path to were the next step of dlc analysis will go
    path_arm_dir = adir + '\\arm_dlc_data'   
    print('checking if: ' + path_arm_dir + 'exists')
    #if the directory hasent been made before we will make it and then run the dlc analysis
    if os.path.exists(path_arm_dir) != True:
        print('making: ' + path_arm_dir)
        os.mkdir(path_arm_dir)
        path_config = 'C:\\Users\\wlwee\\Documents\\python\\fhl_three_target_experiment\\MODEL\\arms-weert-2020-10-18\\config.yaml'
        
        deeplabcut.analyze_videos(path_config,
                                  [vid],
                                  videotype = 'mp4',
                                  save_as_csv = True,
                                  destfolder = path_arm_dir)
        deeplabcut.create_labeled_video(path_config,
                                        [vid],
                                        videotype = 'mp4',
                                        draw_skeleton = 'True',
                                        destfolder = path_arm_dir)
        deeplabcut.plot_trajectories(path_config,
                                     [vid],
                                     videotype = 'mp4',
                                     destfolder = path_arm_dir)
    else:
        print(path_arm_dir + ' already exists, skipping dlc arm model analysis')
    
    #opens the dlc arm model data
    print('getting dlc data')
    path_arm_dlc = glob.glob(path_arm_dir + '\\*.h5')[0]
    print('opening: ' + path_arm_dlc)
    dat_arm_dlc = pd.read_hdf(path_arm_dlc)
    #converts the arm model data frame into a numpy array which is easier to work with
    arm_dlc_numpy = dat_arm_dlc.to_numpy() 
    
    #makes a path to were we will be storing the cleaned arm model data
    path_clean_arm_data_dir = path_arm_dir + '//clean_arm_data' 
    #if the directory hasent been made before this makes it
    print('checking if: ' + path_clean_arm_data_dir + 'exists')
    if os.path.exists(path_clean_arm_data_dir) != True:
        os.mkdir(path_clean_arm_data_dir)
    else:
        print(path_clean_arm_data_dir + ' already exists, not making it again')
    
    #makes four empty dataframes which we will use to store the results from the below script as csvs
    dat_arm_dlc_clean = pd.DataFrame()    
    dat_arm_dlc_clean_real = pd.DataFrame()
    dat_arm_speeds = pd.DataFrame()
    dat_arm_speeds_real = pd.DataFrame()
    
    print()
    for ii in range(0, len(arm_dlc_numpy[0]) - 2,3): #steps left to right through the dataframe by body part
        
        body_part = dat_arm_dlc.columns[ii][1] 
        print(body_part)
        
        print(ii, ii+1, ii+2)
        
        x = []
        y = []
        p = []
        s = 0
        
        for iii in range(0, len(arm_dlc_numpy)): #steps top to bottom through the dataframe for a body part
            x_a = arm_dlc_numpy[iii][ii] #grab the dlc x value
            y_a = arm_dlc_numpy[iii][ii + 1] #grab the dlc y value            
            p_a = arm_dlc_numpy[iii][ii + 2] #grab the probability value
            
            if p_a < p_cutoff: #if p_a is less than p_cutoff we set values to be the last sig detection
                x_a = arm_dlc_numpy[s][ii]
                y_a = arm_dlc_numpy[s][ii + 1]
                p_a = arm_dlc_numpy[s][ii + 2]
                
            else: #p_a > p_cutoff we save the index into a variable s for calling above
                s = iii
                
            p.append(p_a)
            x.append(x_a)
            y.append(y_a)
        
            d_x = []
            d_y = []     
            abs_d_x = []
            abs_d_y = []
            
        for n in range(1, len(x)-1): #time steps through the extracted values from the dataframe, drops 1st and last values            
            speed_x = (x[n] - x[n - 1])
            speed_y = (y[n] - y[n - 1])
            
            '''
            #doing a speed replacement to the last known location ruins it
            #this is because if it jumps a short distance it will get stuck
            if speed_x > 120:
                x[n] = x[n-1]
                speed_x = abs(x[n] - x[n-1])
            if speed_y > 120:
                y[n] = y[n-1]
                speed_y = abs(y[n] - y[n-1])'''
            
            
            d_x.append(speed_x)
            d_y.append(speed_y)             
            abs_d_x.append(abs(speed_x))
            abs_d_y.append(abs(speed_y))
        
        #checking what the speeds are like before filtering
        mean_abs_dx, st_abs_x = st.mean(abs_d_x), st.stdev(abs_d_x)
        mean_abs_dy, st_abs_y = st.mean(abs_d_y), st.stdev(abs_d_y)
        
        mean_dx, st_x = st.mean(d_x), st.stdev(d_x)
        mean_dy, st_y = st.mean(d_y), st.stdev(d_y)
        
        ## some speed filtering step should be here
        ## pure linear interpolation does not fix jumps it 'smoothes' them
                
        # finds indexes where the speeds are outside of mean + 2*stdev d_x
        # does not filter slow speeds only fast speeds
        index_to_interpolate = []
        for i in range(0,len(d_x)):
            
            '''
            #mantle
            if ii <= 6:
                if d_x[i] > mean_dx + 2 * st_x or d_y[i] > mean_dy + 2 * st_y:
                    index_to_interpolate.append(i + 1) '''
            #arms
            if ii > 6:
                #if abs(d_x[i]) > mean_abs_dx + 1.5 * st_abs_x or abs(d_y[i]) > mean_abs_dy + 1.5 * st_abs_y or abs(d_x[i]) > 80 or abs(d_y[i]) > 80:
                if abs(d_x[i]) > 50 or abs(d_y[i]) > 50:
                    index_to_interpolate.append(i + 1) 
        
        interpolate_index_tuple = []
        x_i, y_i = x, y #creates two new lists so we can manipulate x, y lists w/o changing originals
        for i in range(0,len(index_to_interpolate)):
        
            more, less = False, False #two booleans so we can loop until they are both True
            
            #grabs the first before and after indexs
            x1 = x_i[index_to_interpolate[i] - 1] 
            y1 = y_i[index_to_interpolate[i] - 1] 
            x2 = x_i[index_to_interpolate[i] + 1] 
            y2 = y_i[index_to_interpolate[i] + 1] 
            
            itt_1 = index_to_interpolate[i] - 1
            itt_2 = index_to_interpolate[i] + 1
            
            n, m = 1, 1 #grabs starting number of steps and step count
            while more == False and less == False:        
                if i + n < len(index_to_interpolate): #we iterate n forward until we get to a frame that does not need to be iterated
                    if index_to_interpolate[i] + n == index_to_interpolate[i + n]:           
                        #if we have a list iterate 16,17,18,20 and we are at index 0
                        #we step till index 2 then stop there
                        #we interpolate till the x2 index 18 + 1
                        x2 = x_i[index_to_interpolate[i + n] + 1]
                        y2 = y_i[index_to_interpolate[i + n] + 1] 
                        itt_2 = index_to_interpolate[i + n] + 1
                        n = n + 1 
                    else:
                        more = True
                else:
                    more = True
                
                if i - m > 0: #same as above but going backwards
                    if index_to_interpolate[i] - m == index_to_interpolate[i - m]:
                        x1 = x_i[index_to_interpolate[i - m] - 1]
                        y1 = y_i[index_to_interpolate[i - m] - 1]
                        itt_1 = index_to_interpolate[i - m] - 1
                        m = m + 1
                    else:
                        less = True
                else:
                    less = True
            
            interpolate_index_tuple.append([itt_1, itt_2, i-m, i+n])
            #n equals steps forward from the bad x value to the x value we want to iterate to
            #m equals steps backwards from the bad x value to the x value we want to iterate to
            #steps = n + m
            at = m/(n+m)
            
            '''
            #incase of errors uncomment this section
            print()
            print(ii, iii)
            print(x1,y1)
            print(x2,y2)
            print(n+m,m,n,at)
            '''
            #see interpolate def above for behavior
            x_inter, y_inter = interpolate(float(x1),float(x2),float(y1),float(y2),at)
            #rewrite x_i for the interpolate index to x_inter
            x_i[index_to_interpolate[i]] = x_inter
            y_i[index_to_interpolate[i]] = y_inter
            '''
            print(x_inter, y_inter)
            print()        
            '''
            
        d_xi, d_yi = [], [] #new variables to hold interapolation speeds so we can investigate them
        
        for n in range(1, len(x_i)-1): 
            
            speed_xi = (x_i[n] - x_i[n - 1])
            speed_yi = (y_i[n] - y_i[n - 1])
            d_xi.append(speed_xi)
            d_yi.append(speed_yi)
        
        x_ri, y_ri = [], []
        
        for i in range(0, len(x_i)): #get the 'real' x, y values by shifting relative to the x1, y1 crop parameters
            x_ri_a = x_i[i] + crop_x1y1_parameters[i][0]
            y_ri_a = y_i[i] + crop_x1y1_parameters[i][1]
            
            x_ri.append(x_ri_a)
            y_ri.append(y_ri_a)
        
        d_xi_r, d_yi_r = [], []
        
        for n in range(1, len(x_ri)-1):             
            speed_xi_r = (x_ri[n] - x_ri[n - 1])
            speed_yi_r = (y_ri[n] - y_ri[n - 1])
            d_xi_r.append(speed_xi_r)
            d_yi_r.append(speed_yi_r)
        
        #here we calculate speed means and standard deviations for the filtered data before and after it is returned to the 'real' reference frame
        mean_dxi, st_xi = st.mean(d_xi), st.stdev(d_xi)
        mean_dyi, st_yi = st.mean(d_yi), st.stdev(d_yi)
        
        mean_dxi_r, st_xi_r = st.mean(d_xi_r), st.stdev(d_xi_r)
        mean_dyi_r, st_yi_r = st.mean(d_yi_r), st.stdev(d_yi_r)
        
        print('number interpolated frames ' + str(len(index_to_interpolate)))
        print('unfiltered speed ' + body_part + '_x: ' + str(round(mean_dx,3)) + ' +- ' + str(round(st_x,3)))
        print('filtered speed ' + body_part + '_x: ' + str(round(mean_dxi,3)) + ' +- ' + str(round(st_xi,3)))
        print('filtered real speed ' + body_part + '_x: ' + str(round(mean_dxi_r,3)) + ' +- ' + str(round(st_xi_r,3)))
        print('unfiltered speed ' + body_part + '_y: ' + str(round(mean_dy,3)) + ' +- ' + str(round(st_y,3)))
        print('filtered speed ' + body_part + '_y: ' + str(round(mean_dyi,3)) + ' +- ' + str(round(st_yi,3)))
        print('filtered real speed ' + body_part + '_y: ' + str(round(mean_dyi_r,3)) + ' +- ' + str(round(st_yi_r,3)))
        p_test = p[2:]
        print('p min ' + str(round(min(p_test),5)))
        print()
            
        #creates the dataframes that will store our filtered positional data
        #octopus reference frame and real reference frame
        #column names are tuples
        dat_arm_dlc_clean[dat_arm_dlc.columns[ii]] = x_i
        dat_arm_dlc_clean[dat_arm_dlc.columns[ii + 1]] = y_i
        dat_arm_dlc_clean[dat_arm_dlc.columns[ii + 2]] = p
        
        dat_arm_dlc_clean_real[dat_arm_dlc.columns[ii]] = x_ri
        dat_arm_dlc_clean_real[dat_arm_dlc.columns[ii + 1]] = y_ri
        dat_arm_dlc_clean_real[dat_arm_dlc.columns[ii + 2]] = p
        
        #creates the dataframes that will store our filtered speed data
        #octopus reference frame and real reference frame
        #column names are tuples
        dat_arm_speeds[dat_arm_dlc.columns[ii]] = d_xi
        dat_arm_speeds[dat_arm_dlc.columns[ii + 1]] = d_yi   
        
        dat_arm_speeds_real[dat_arm_dlc.columns[ii]] = d_xi_r
        dat_arm_speeds_real[dat_arm_dlc.columns[ii + 1]] = d_yi_r
        
    path_dat_arm_dlc_clean_csv = path_clean_arm_data_dir + '\\octo_ref_clean_arm_data.csv'
    dat_arm_dlc_clean.to_csv(path_dat_arm_dlc_clean_csv)
    
    path_dat_arm_dlc_clean_real_csv = path_clean_arm_data_dir + '\\octo_real_ref_clean_arm_data.csv'
    dat_arm_dlc_clean_real.to_csv(path_dat_arm_dlc_clean_real_csv)
    
    path_dat_arm_speeds_csv = path_clean_arm_data_dir + '\\octo_ref_speed.csv'
    dat_arm_speeds.to_csv(path_dat_arm_speeds_csv)
    
    path_dat_arm_speeds_real_csv = path_clean_arm_data_dir + '\\octo_ref_speed_real.csv'
    dat_arm_speeds_real.to_csv(path_dat_arm_speeds_real_csv)
    
    ## below we are going to make a video using our filtered data to draw points and lines upon it
    ## we do this to investigate what our filtered predictions look like
    numpy_arm_dlc_clean = dat_arm_dlc_clean.to_numpy()
    numpy_arm_dlc_clean_real = dat_arm_dlc_clean_real.to_numpy()
    numpy_arm_speed = dat_arm_speeds.to_numpy()
    numpy_arm_speed_real = dat_arm_speeds_real.to_numpy()
    vid_dlc_labeled = glob.glob(path_arm_dir + '\\*.mp4')[0]
    video_clean_labels_name = path_clean_arm_data_dir + '\\clean_labels_vid.mp4'
    video_clean_labels_hull_name = path_clean_arm_data_dir + '\\clean_labels_hull_vid.mp4'
    
    #open the three target location .csv
    target_location_path = os.path.dirname(os.path.dirname(adir)) + '\\three_target_location.csv'
    if os.path.exists(target_location_path) != True:
        print(target_location_path + ' does not exist!')
    else: 
        print('opening ' + target_location_path)
        dat_target = pd.read_csv(target_location_path)
    
    #turn the dat_target dataframe into a numpy array
    numpy_target = dat_target.to_numpy()
    food_location_xy = []
    #loop left to right through the first row of the array and find which target has food
    for i in range(0, len(numpy_target[0]),3):
        print(numpy_target[0][i],numpy_target[0][i+1],numpy_target[0][i+2])
        if numpy_target[0][i+2] == 1:
            food_location_xy.append([numpy_target[0][i],numpy_target[0][i+1]])
    food_location_xy = food_location_xy[0]
    
    cap = cv2.VideoCapture(vid)
    cap_dlc = cv2.VideoCapture(vid_dlc_labeled)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    fs = 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video = cv2.VideoWriter(video_clean_labels_name, 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            7.0, 
                            (width, height))
    
    video_hull = cv2.VideoWriter(video_clean_labels_hull_name, 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    7.0, 
                                    (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    hull_list = []
    hull_list_rotated = []
    l_eye_list, r_eye_list, mean_eye_list = [] ,[], []
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret_dlc, frame_dlc = cap_dlc.read()        
        frame_edit = frame
        
        xy_lst = numpy_arm_dlc_clean[fs].tolist() #grab list of coordinates
        del xy_lst[2::3] #reshape the list to remove p-values
        hull_points = []
        for i in range(0, len(xy_lst), 2): 
             hull_points.append([xy_lst[i], xy_lst[i+1]]) #turn the list of body parts into a coordinate list
        
        #give the coordinate list to the convexHull function, saves a list of coordinates
        hull = convexHull(hull_points, len(hull_points))
        #make new pts that is in the format opencv fillPoly wants
        pts = np.array(hull, np.int32) 
        pts = pts.reshape((-1,1,2))  
        
        #create a copy of frame_edit to overlay on frame edit
        overlay = frame_edit.copy() 
        alpha = 0.3 #transparency for the frame_edit overlay
        cv2.fillPoly(overlay,[pts],(100,255,255)) #put a polygon on the overlay
        frame_edit = cv2.addWeighted(overlay, alpha, frame_edit, 1 - alpha, 0) #put the overlay on frame_edit  
        
        #save the hull
        hull_list.append(hull)
        
        #convex hull octopus frame of view re-orientation        
        
        #save the eye coordinates
        le_xy = [numpy_arm_dlc_clean[fs][0], numpy_arm_dlc_clean[fs][1]]
        l_eye_list.append(le_xy)
        re_xy = [numpy_arm_dlc_clean[fs][3], numpy_arm_dlc_clean[fs][4]]
        r_eye_list.append(re_xy)
        
        #get mean eye coordinates
        mean_e_xy = [st.mean([le_xy[0], re_xy[0]]), st.mean([le_xy[1], re_xy[1]])]
        mean_eye_list.append(mean_e_xy)
        
        #move left eye and right eye to the origin as we can get angle to rotate the hull
        l_xy_o = [le_xy[0]-mean_e_xy[0], le_xy[1]-mean_e_xy[1]]
        r_xy_o = [re_xy[0]-mean_e_xy[0], re_xy[1]-mean_e_xy[1]]
        
        #atan2 gives negative and positive radian values relative to the positive x-axis
        theta = math.atan2(l_xy_o[1],l_xy_o[0])
        #theta_r = math.atan2(r_xy_o[1],r_xy_o[0])
        
        #we will rotate the right eye to the positive x-axis and left eye to the negative x-axis
        le_re_reorientated_list = []
        le_xy_r = rotate([0,0], l_xy_o, (math.pi-theta))
        re_xy_r = rotate([0,0], r_xy_o, (math.pi-theta))
        le_re_reorientated_list.append([le_xy_r,re_xy_r])
        
        #since we are dealing with big floats for my sake I set the eyes to zero
        #the eyes so far are not helpful
        if round(le_xy_r[1],5) > 0 or round(re_xy_r[1],5) > 0:
            print(str(fs, le_xy_r[1], re_xy_r[1]) + ' eye rotate error theta_l < 0 and l_xy_o[0] < r_xy_o[0]')
        else:
            le_xy_r[1] = 0.0
            re_xy_r[1] = 0.0
        
        #here we rotate the hull
        rotated_hull = []
        for hull_cord in hull:
            ho_xy = [hull_cord[0]- mean_e_xy[0], hull_cord[1]- mean_e_xy[1]]
            ho_xy_r = rotate([0,0], ho_xy, (math.pi-theta))
            ho_xy_r = [ho_xy_r[0] + width/2, ho_xy_r[1] + height/2]
            rotated_hull.append(ho_xy_r)
            
        hull_list_rotated.append(rotated_hull)
        
        '''
        frame_hull_unrotated = np.zeros((width,height,3),np.uint8)
        pts = np.array(hull, np.int32) 
        pts = pts.reshape((-1,1,2)) 
        cv2.fillPoly(frame_hull_unrotated,[pts],(255,255,255))'''
        
        frame_hull_rotated = np.zeros((width,height,3),np.uint8)
        pts = np.array(rotated_hull, np.int32) 
        pts = pts.reshape((-1,1,2)) 
        cv2.fillPoly(frame_hull_rotated,[pts],(255,255,255))
        
        cv2.circle(frame_hull_rotated,
                   (int(width/2),int(height/2)),
                   1,
                   (0,0,0))
        
        ##get the real eye coordinates so we can point at the food
        #save the eye coordinates
        le_xy_real = [numpy_arm_dlc_clean_real[fs][0], numpy_arm_dlc_clean_real[fs][1]]
        re_xy_real = [numpy_arm_dlc_clean_real[fs][3], numpy_arm_dlc_clean_real[fs][4]]
        #get real mean eye coordinates
        mean_e_xy_r = [st.mean([le_xy_real[0], re_xy_real[0]]), st.mean([le_xy_real[1], re_xy_real[1]])]
        
        food_location_xy_r = [food_location_xy[0] - mean_e_xy_r[0], food_location_xy[1] - mean_e_xy_r[1]]
        
        food_location_xy_r_r = rotate([0,0], food_location_xy_r, (math.pi-theta)) 
        food_location_xy_r_r_o = [food_location_xy_r_r[0] + width/2, food_location_xy_r_r[1] + height/2]
        
        #get the mean speed of the eyes 
        if fs > 0 and fs < frame_count - 2:
            speed_le_xy = [numpy_arm_speed_real[fs][0],numpy_arm_speed_real[fs][1]]
            speed_re_xy = [numpy_arm_speed_real[fs][2],numpy_arm_speed_real[fs][3]]
            
            speed_em_xy = [st.mean([speed_le_xy[0],speed_re_xy[0]]),
                           st.mean([speed_le_xy[1],speed_re_xy[1]])]           
            speed_em_xy_rotated = rotate([0,0], speed_em_xy, (math.pi-theta))
            
            sxy = [int(10*speed_em_xy_rotated [0] + width/2),int(10*speed_em_xy_rotated [1] + height/2)]
            
            print(food_location_xy_r_r)
            cv2.arrowedLine(frame_hull_rotated,
                            (int(width/2),int(height/2)),
                            (int(food_location_xy_r_r_o[0]),int(food_location_xy_r_r_o[1])),
                            (200,100,0),thickness=2)
            
            cv2.arrowedLine(frame_hull_rotated,
                     (int(width/2),int(height/2)),
                     (sxy[0],sxy[1]),
                     (100,200,0),thickness=3)
            
            speed_text = '('+ str(round(speed_em_xy_rotated[0],2)) + ',' + str(round(speed_em_xy_rotated[1],2)) + ')'
            cv2.putText(frame_hull_rotated, 
                            speed_text,
                            (sxy[0],sxy[1]),
                            font,
                            0.25,
                            (100,200,0))
        
        #draw whitish grey lines using body_part_line_list
        for part in body_part_line_list:
            xy_l0 = int(numpy_arm_dlc_clean[fs][part[0]*3]), int(numpy_arm_dlc_clean[fs][part[0]*3 + 1])
            xy_l1 = int(numpy_arm_dlc_clean[fs][part[1]*3]), int(numpy_arm_dlc_clean[fs][part[1]*3 + 1])
            cv2.line(frame_edit, xy_l0, xy_l1, (255/2,255/2,255/2), 1)
        
        #draw color coded body parts
        #goes left to right through the numpy array by steps of 3
        ss = 0
        for i in range(0, len(numpy_arm_dlc_clean[fs]), 3):
            xy_c = int(numpy_arm_dlc_clean[fs][i]) , int(numpy_arm_dlc_clean[fs][i + 1])
            
            col1 = int(255 - 255/len(numpy_arm_dlc_clean[fs])*i)
            col2 = int(255/len(numpy_arm_dlc_clean[fs])*i)            
                                              
            cv2.circle(frame_edit, xy_c, 3, (col1,
                                             0,
                                             col2),
                                             -1)
            if fs > 0 and fs < frame_count - 2:
                speed_xt = numpy_arm_speed[fs][i - ss]
                speed_yt = numpy_arm_speed[fs][i - ss + 1]
                
                speed_xy = str(round(math.sqrt(speed_xt**2 + speed_yt**2),1))
                
                cv2.putText(frame_edit, 
                            speed_xy,
                            xy_c,
                            font,
                            0.25,
                            (255/2,255/2,255/2))   
            
                ss = ss + 1                                             
        
        '''cv2.putText(frame_edit,
                    'speed ')'''
        
        #cv2.imshow('FrameClear',frame)
        cv2.imshow('FrameEdit',frame_edit)
        cv2.imshow('FrameDLC',frame_dlc)
        #cv2.imshow('FrameUnrotatedHull',frame_hull_unrotated)
        cv2.imshow('FrameRotatedHull',frame_hull_rotated)
        
        video.write(frame_edit)
        video_hull.write(frame_hull_rotated)
        
        cv2.waitKey(25)
        fs = fs + 1
        if fs == frame_count: 
            break
    
    video_hull.release()
    video.release()    
    cap_dlc.release()
    cap.release()
    cv2.destroyAllWindows()

#list of animals to be investigated
animal_list = ['khorne',
               'nips',
               'slaanesh',
               'nurgle']

#root dir where all the animal video directories are stored
#directory structure expected goes...
#animal > trial days (month_day_trial_##) > all trials for a day (month_day_arrayorder_eatnoeat_##)
path_animal_vids_dir_root = r'W:\FHL_FLUME\three_target_experiment\eat_videos_food_to_mouth_five_min_back'

#gets a lit of the different animal root dirs
path_animal_vids_dirs = []
for animal_vid in animal_list:
    path_animal_vids_dirs.append(path_animal_vids_dir_root + '\\' + animal_vid)
    
#gets the path to all of the trial days for the different animals
path_dlc_csv = []
for an_animal in path_animal_vids_dirs:
    path_dlc_csv = path_dlc_csv + glob.glob(an_animal + '\\*\\*\\dlc_data\\*.csv')

## if you want to copy videos in a dir so they are all in one spot per animal turn true
copy_videos = False
## set where the directory is that you want to copy the videos to
copy_videos_to_dir = r'W:\FHL_FLUME\arm_model_videos'

'''
#turn this code back on if you only want to analyze one animal at a time

#this path determines which animal dir you are looking at
path_animal_vids_dir = r'W:\FHL_FLUME\three_target_experiment\eat_videos_food_to_mouth_five_min_back\khorne'
#this grabs the dlc csv files
path_dlc_csv = glob.glob(path_animal_vids_dir + '\\*\\*\\dlc_data\\*.csv') ## limit of this is that this wont show us empty dirs
'''

body_part_line_list = [[0,1],[0,4],[1,4],[6,2],[2,3],[3,4],[4,5],[5,6], #mantle
                       [7,8],[8,9],[10,11],[11,12],[13,14],[14,15],[16,17],[17,18],[19,20],[20,21],[22,23],[23,24],[25,26],[26,27],[28,29],[29,30], #arm length
                       [9,12],[12,15],[15,18],[18,21],[21,24],[24,27],[27,30],[30,9]] #arm tips
p_cutoff = 0.6

## if you do not want to capture frames where the octopus is on food turn true
remove_food_values = False
## set parameters for box around the food
xf = [1220, 1330]
yf = [320, 850]

## if you do not want to capture frames where the octopus is in the den turn true
remove_den_values = False
## set parameters for box around the den
xd = [400, 580]
yd = [530, 650]

#how many seconds you want to look back from your 'touch' frame of interest
seconds_to_look_back = 120
#determine if you want to go back from mouth or arm touch
#arm touch = 0, mouth touch = 1
touch_of_interest = 1

#set crop size, octopus has a crop_soze x crop_size box cropped around the mid point of its eyes
crop = 600

for acsv in path_dlc_csv:
    
    #get eat no eat
    eat_noeat = os.path.basename(os.path.dirname(os.path.dirname(acsv))).split('_')[-2]
    
    #get array format
    array_format = os.path.basename(os.path.dirname(os.path.dirname(acsv))).split('_')[-3]
    
    #go back from the dlc csv to get the other csvs
    touch_csv_path = os.path.dirname(os.path.dirname(acsv)) + '\\three_target_touch.csv'
    path_frames_csv = os.path.dirname(os.path.dirname(acsv)) + '\\' + os.path.basename(os.path.dirname(os.path.dirname(acsv))) + '.csv'
    
    #checks if your csvs exist if not the loop skips them and continues
    if os.path.exists(touch_csv_path) == True and os.path.exists(path_frames_csv) == True and os.path.exists(acsv):
        dat_dlc = pd.read_csv(acsv)
        dat_touch = pd.read_csv(touch_csv_path)
        dat_frames = pd.read_csv(path_frames_csv)
    else:
        print('missing csv data')
        continue
    
    #path to a new dir that will hold the data for the arm model and analysis
    arm_frame_stor_dir = os.path.dirname(os.path.dirname(acsv)) + '\\arm_frames_data'
    #checks if the arm dir exists and if not makes it
    if os.path.exists(arm_frame_stor_dir) != True:
        os.mkdir(arm_frame_stor_dir)
     
    which_animal =  arm_frame_stor_dir.split('\\')[-4]   
    
    #gets the path to the 'big' frames which we will then be cropping from
    frame_paths = dat_frames['frame_paths']
    #gets the floats from the 'big' frame so we can use closest() to get which frame to start looking back from
    frame_floats = dat_frames['frame_floats']
    
    #get the index for the touch of interest and determine how many frames to look back
    #number of frames you can look back is limited by the length of the analyzed video
    index_closest = closest(frame_floats, iterate_touch_csv(dat_touch, touch_of_interest))
    index_look_back = closest(frame_floats, (index_closest[1] - seconds_to_look_back))
    
    #gets the path to the sequence of images the video represents
    #this is for future cropping
    path_frame_seq = frame_paths[index_look_back[0]:index_closest[0]+1].tolist()
    frame_floats_seq = frame_floats[index_look_back[0]:index_closest[0]+1].tolist()
    
    #get fps
    fps_seq = get_fps(frame_floats_seq)
    
    #paths for where your cropped video and reduce dlc data will go
    arm_frame_data_path = arm_frame_stor_dir + '\\' + str(seconds_to_look_back) + '_sb_' + str(touch_of_interest) + '_touch_' + array_format + '_' + eat_noeat 
    if os.path.exists(arm_frame_data_path) != True:
        os.mkdir(arm_frame_data_path)
    else:
        print(arm_frame_data_path + ' already exists')
    arm_frame_data_path = arm_frame_data_path + '\\' + os.path.basename(arm_frame_data_path)
    arm_frame_vid_path = arm_frame_data_path + '.mp4'
    arm_frame_csv_path = arm_frame_data_path + '.csv'
    
    #subset the dlc data you analyzed for the index you are interested minus how many seconds to look back
    dat_dlc_interesting_seq = dat_dlc[index_look_back[0]+2:index_closest[0]+3]
    dat_dlc_interesting_seq['img_paths'] = ''
    dat_dlc_interesting_seq = dat_dlc_interesting_seq.assign(img_paths=path_frame_seq)
    
    #get the mean values from the reduced dlc csv for cropping
    mean_values = get_mean_pixel_value(dat_dlc_interesting_seq)
    
    #store the mean values into the reduced dlc dataframe
    dat_dlc_interesting_seq['mean_x'] = ''
    dat_dlc_interesting_seq = dat_dlc_interesting_seq.assign(mean_x=mean_values[0])
    dat_dlc_interesting_seq['mean_y'] = ''
    dat_dlc_interesting_seq = dat_dlc_interesting_seq.assign(mean_y=mean_values[1])
    
    #make a plot of the mean values from the reduced dlc dataframe
    make_many_colored_plot(mean_values[0],
                           mean_values[1],
                           dat_dlc_interesting_seq['img_paths'].iloc[0],
                           title = '',
                           plot_save_path = arm_frame_data_path + 'center_plot.png',
                           save_plots = True)
    
    index_test = []
    for i in range(0,len(path_frame_seq)):
        
        x_test = mean_values[0][i]
        y_test = mean_values[1][i]
        
        if x_test == 0 and y_test == 0:
            index_test.append(i)
               
    print('removing ' + str(len(index_test)) + ' bad dlc predictions from lists')
    x_values_removed = [i for j, i in enumerate(mean_values[0]) if j not in index_test]
    y_values_removed = [i for j, i in enumerate(mean_values[1]) if j not in index_test]
    frames_removed = [i for j, i in enumerate(path_frame_seq) if j not in index_test]
    
    print('removing ' + str(len(index_test)) + ' bad dlc predictions from dat_dlc_interesting_seq')   
    dat_dlc_interesting_seq = dat_dlc_interesting_seq.drop(dat_dlc_interesting_seq.index[index_test])
    
    if remove_den_values == True:
        index_test = []
        for i in range(0, len(frames_removed)):
            
            x_test = x_values_removed[i]
            y_test = y_values_removed[i]
            
            if xd[0] < x_test and x_test < xd[1] and yd[0] < y_test and y_test < yd[1]:
                index_test.append(i)
        
        print('removing ' + str(len(index_test)) + ' in den frames')
        x_values_removed = [i for j, i in enumerate(x_values_removed) if j not in index_test]
        y_values_removed = [i for j, i in enumerate(y_values_removed) if j not in index_test]
        frames_removed = [i for j, i in enumerate(frames_removed) if j not in index_test]
        
        print('removing ' + str(len(index_test)) + ' den frames from dat_dlc_interesting_seq')
        dat_dlc_interesting_seq = dat_dlc_interesting_seq.drop(dat_dlc_interesting_seq.index[index_test]) 
          
    if remove_food_values == True:
        index_test = []
        for i in range(0, len(frames_removed)):
            
            x_test = x_values_removed[i]
            y_test = y_values_removed[i]
            
            if xf[0] < x_test and x_test < xf[1] and yf[0] < y_test and y_test < yf[1]:
                index_test.append(i)
        
        print('removing ' + str(len(index_test)) + ' on food')        
        x_values_removed = [i for j, i in enumerate(x_values_removed) if j not in index_test]
        y_values_removed = [i for j, i in enumerate(y_values_removed) if j not in index_test]
        frames_removed = [i for j, i in enumerate(frames_removed) if j not in index_test]
        
        print('removing ' + str(len(index_test)) + ' food frames from dat_dlc_interesting_seq')
        dat_dlc_interesting_seq = dat_dlc_interesting_seq.drop(dat_dlc_interesting_seq.index[index_test])
    
    mean_values_removed = [x_values_removed, y_values_removed]
    
    #make a plot of the mean values from the reduced dlc dataframe
    make_many_colored_plot(x_values_removed,
                           y_values_removed,
                           dat_dlc_interesting_seq['img_paths'].iloc[0],
                           title = '',
                           plot_save_path = arm_frame_data_path + 'center_plot_zeros_removed.png',
                           save_plots = True)
    
    #makes cropped videos centered on the octopus
    crop_parameter_list = make_cropped_video(mean_values_removed,
                                               frames_removed,
                                               crop,
                                               arm_frame_vid_path,
                                               fps_seq)
    
    print('sticking crop parameters into dat_dlc_interesting_seq')
    dat_dlc_interesting_seq['crop_parameters'] = ''
    dat_dlc_interesting_seq = dat_dlc_interesting_seq.assign(crop_parameters=crop_parameter_list)
    
    print('getting paths for video copies')
    video_copy_head = which_animal + '_' + arm_frame_stor_dir.split('\\')[-2] + '_' + os.path.basename(arm_frame_vid_path)
    video_copy_animal_dir = copy_videos_to_dir + '\\' + which_animal
    video_copy_name = video_copy_animal_dir + '\\' + video_copy_head
    
    if os.path.exists(video_copy_animal_dir) != True and copy_videos == True:
        os.mkdir(video_copy_animal_dir)
        
    if copy_videos == True:
        copyfile(arm_frame_vid_path, video_copy_name)
    
    #save the reduced dlc dataframe
    print('saving: ' + arm_frame_csv_path)
    dat_dlc_interesting_seq.to_csv(arm_frame_csv_path)
    
    print('analyzing and cleaning data in ' + os.path.dirname(os.path.dirname(acsv)))
    analyze_and_clean_arm_dat(dat_mantle_dlc = dat_dlc_interesting_seq,
                              adir = os.path.dirname(arm_frame_csv_path),
                              body_part_line_list = body_part_line_list,
                              pcutoff = 0.6,
                              play_back_speed = 10)
    
        
        
    
    
    
    
    
    
    
    
    
    
    