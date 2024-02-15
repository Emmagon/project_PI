
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import csv
import random
import subprocess
import shutil
import os
import sys
from scipy.interpolate import interp1d
from datetime import datetime
import time
import threading
from collections import deque
from tensorflow import gather_nd
import tensorflow as tf

def update_q(q, index, value):
    indices = tf.constant([[0, index]])
    updates = tf.constant([value], dtype=tf.float32)
    return tf.tensor_scatter_nd_update(q, indices, updates)
def copy_and_rename_files(episode, count):
    #count=episode+1+count*0.01
    user_name = r"C:\Users\emago"
    source_folder = user_name + r"\Desktop\PI_MARL\H-board\DQN\Design1.emc\PI-1/Power_GND"  # 替换为源文件夹的路径
    file_list = ["1-PIPinZ_IC1.csv", "1-PIPinZ_IC1.srdb"]  # 要复制的文件列表
    new_names = [str(episode+1)+"_"+str(count)+"_1-PIPinZ_IC1.csv", str(episode+1)+"_"+str(count)+"_1-PIPinZ_IC1.srdb"]  # 新文件名列表

    folder_path= user_name + r"\Desktop\PI_MARL\H-board\DQN\result_dqn"# 替换为目标文件夹的路径
    folder_name = "record_impedance_dqn"
    target_folder = os.path.join(folder_path, folder_name)
    if not os.path.exists(target_folder):
      os.makedirs(target_folder)

    for src_file, new_name in zip(file_list, new_names):
        src_path = os.path.join(source_folder, src_file)
        dst_path = os.path.join(target_folder, new_name)
        shutil.copy2(src_path, dst_path)  # 使用copy2以保留文件的元数据
# initial matrix
def initial_matrix(array,row_num,col_num):
    matrix = np.full((row_num, col_num), -1)

    for i, val in enumerate(array):
        row = (val - 1) // col_num
        col = (val - 1) % col_num
        matrix[row, col] = 0
    return matrix

# define statespace
def possible_state_array(initial_layout, num_type, num_pos):
    decap_type = np.array(list(np.ndindex((num_type + 1,) * num_pos)))
    state_list = []
    for pem in decap_type:
        possible_state_layout = np.copy(initial_layout)
        # Replace all values greater than 0 with the values in pem to create a new array object
        state_list.append(possible_state_layout)
        possible_state_layout[possible_state_layout >= 0] = pem
    state_list = np.array(state_list)
    return state_list
def next_state_step(state, action,state_einheit):
    tmp_next_state = np.copy(state)
    tmp_next_state[state_einheit-1] = action[0]
    tmp_next_state[2*state_einheit-1] = action[1]
    for i in range(state_einheit):
        for j in range(0, state_einheit - i - 1):
            if tmp_next_state[j] < tmp_next_state[j + 1]:
                tmp_next_state[j], tmp_next_state[j + 1] = tmp_next_state[j + 1], tmp_next_state[j]
                tmp_next_state[j + state_einheit], tmp_next_state[j + state_einheit+1] = tmp_next_state[j + state_einheit+1], tmp_next_state[j + state_einheit]
    return tmp_next_state
# define actionspace
def possible_action(pos_list, type_list):
    action_list = []
    for p,pos in enumerate(pos_list):
        for t,type in enumerate(type_list):
                action_list.append([pos, type])
    return action_list
def get_freqency_and_impedance(current_impedance_list):
    i = 0
    while i < len(current_impedance_list):
        sublist = current_impedance_list[i]
        j = 0
        while j < len(sublist):
            try:
                sublist[j] = float(sublist[j])
                j += 1
            except ValueError:
                del current_impedance_list[i]
                break
        else:
            i += 1
    del current_impedance_list[0]
    num_rows = len(current_impedance_list)
    # print(num_rows)

    impedance_values = [row[1] for row in current_impedance_list]
    frequency = [row[0] for row in current_impedance_list]
    return frequency, impedance_values,num_rows
def calculate_mse(num_rows, impedance_values, frequency, fmin, fmax, fmittel, Zmin, Zmax):
    target_values = []
    simulated_values = []
    tmp = 0
    tolerance = 0
    count = 0
    for i in range(num_rows):
        target_value = target_impedance(frequency[i], fmin, fmax, fmittel, Zmin, Zmax)
        if impedance_values[i] > target_value:
            target_values.append(target_value)
            simulated_values.append(impedance_values[i])
            count += 1

    simulated_array = np.array(simulated_values)
    target_array = np.array(target_values)
    return count, tmp, tolerance,target_array,simulated_array

def calculate_point(num_rows, impedance_values, frequency, fmin, fmax, fmittel, Zmin, Zmax,tolerance_fre):
    count = 0  # how many points satisfy the target impedance
    tmp = 0
    tolerance = 0
    for i in range(num_rows):
        if impedance_values[i] <= target_impedance(frequency[i], fmin, fmax, fmittel, Zmin, Zmax):
            # datatype
            count += 1
        if frequency[i] >= tolerance_fre:
            tmp += 1
            if impedance_values[i] > target_impedance(frequency[i], fmin, fmax, fmittel, Zmin, Zmax):
                tolerance += 1
    return count, tmp, tolerance
# find position and type of decap in action
def decap_in_action(action,num_type,pos_list):
    position = 1
    for i, row in enumerate(action):
        for j, value in enumerate(row):
            if value in range(1, num_type + 1):
                position = pos_list[position - 1]
                return position, value
            position += 1

# copy batch file
def copy_batch_file(destination_file,source_file):
    # source_file = "/content/batch.peb"
    # destination_file = "/content/tmp_batch.peb"

    # delete the last tmp_batch.peb
    try:
        os.remove(destination_file)
    except:
        None
    # use shutil to copy and rename the file
    shutil.copyfile(source_file, destination_file)
    # check
    if os.path.isfile(destination_file):
        None
    else:
        sys.exit("Error copying file")

def find_pos_type(matrix,pos_list,col_num):
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            if col > 0:
                return pos_list[i*col_num + j], col

# load batch file
def load_batch_file(destination_file):
    user_name = r"C:\Users\emago"
    if os.path.exists(user_name + r'\Desktop\PI_MARL\H-board\DQN\Design1.emc\Design1.rlk'):
        os.remove(user_name + r'\Desktop\PI_MARL\H-board\DQN\Design1.emc\Design1.rlk')
        print("The '.rlk' file exists and will be deleted!")
    # else:
    # print("OK:NO .rlk")

    Zukenpath = r"C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe"
    PEBPath = destination_file
    ERFPath = user_name + r"\Desktop\PI_MARL\H-board\DQN\Design1.emc\Design1.erf"

    try:
        subprocess.check_output([Zukenpath, '--batch', PEBPath,'--batch-auto-exit', ERFPath])
    except subprocess.CalledProcessError as err:
        print(err)
    except subprocess.TimeoutExpired as err:
        print(err)

# define target impedance
def target_impedance(f,fmin,fmax,fmittel,Zmin,Zmax):
    if f >= fmin and f < fmittel:
        return Zmin
    if f <= fmax and f >= fmittel:
        tmp_x=[fmittel,fmax]
        tmp_y = [Zmin, Zmax]
        function = interp1d(tmp_x, tmp_y, kind='linear')
        return function(f)
    else:
        print("falsch range of frequency")
        return None