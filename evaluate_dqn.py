
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
from scipy.interpolate import griddata

def plot_graph(reward_average,episode_num, decaps_num, reward_list, file_path, episode,current_date):

    #1. plot the number of decaps
    plt.plot(range(len(decaps_num)), decaps_num, linewidth=0.5, marker='o', markersize=0.6)
    plt.xlabel('number of episodes')
    plt.ylabel('number of decaps')
    plt.title(' Number of decaps assigned to the optimized PDN DQN')
    plt.grid(True)
    plt.savefig(file_path+r'\decap_num_'+str(episode+1)+'_times_'+current_date+'.png')
    plt.show()

    #reward
    plt.plot(range(len(reward_list)), reward_list, linewidth=0.5, marker='o', markersize=0.6)
    plt.xlabel('number of episodes')
    plt.ylabel('value of reward')
    plt.title(' final reward assigned to the optimized PDN')
    plt.grid(True)
    plt.savefig(file_path + '\Reward_' +str(episode + 1) + '_times_' + current_date + '.png')
    plt.show()

    plt.plot(range(len(reward_average)), reward_average, linewidth=0.5, marker='o', markersize=0.6)
    plt.xlabel('number of episodes')
    plt.ylabel('average reward')
    plt.title(' average rewar')
    plt.grid(True)
    plt.savefig(
        file_path + '\Reward_average' + str(episode + 1) + '_times_' + current_date + '.png')
    plt.show()

def record_csv(each_step,each_pos,each_type,each_state,file_path,episode,is_double):
    header = ['Step','selected Position','selected Type','State']
    rows = zip(each_step,each_pos,each_type,each_state)
    with open(file_path+'\Record_'+str(episode+1)+'_times_'+is_double+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

def result_csv(reward_average,episode_num,result_list,reward_list,decaps_num,file_path,episode):
    header = ['Episode','Reward']
    rows = zip(episode_num,reward_list)
    with open(file_path+'\Reward_'+str(episode+1)+'_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    header = ['Episode', 'Result']
    rows = zip(episode_num,result_list)
    with open(file_path+'\Result_'+str(episode+1)+'_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    header = ['Episode', 'Decap_num']
    rows = zip(episode_num,decaps_num)
    with open(file_path+'\Decap_'+str(episode+1)+'_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)



    header=['Episode','Reward of average']
    rows=zip(episode_num,reward_average)
    with open(file_path+'\Reward_average_'+str(episode+1)+'_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


