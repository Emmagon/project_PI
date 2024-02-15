import numpy as np
import xml.etree.ElementTree as ET
import random
import csv
import os
import sys
from datetime import datetime
import threading
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten
import evaluate_dqn as eva
import help_dqn as fh
from collections import deque

##########################################################################################
#prepare matrix, position, type parameters
##########################################################################################
state_einheit=8
state_num = int(2*state_einheit)

pos_list = np.array([1,2,3,4,5,6,7,8,9,10])
num_pos = len(pos_list)

# path and file name of batch file
user_name = r"C:\Users\emago"
source_file = user_name + r"\Desktop\PI_MARL\H-board\DQN\Batch_10Decaps_Values.peb"
destination_file = user_name + r"\Desktop\PI_MARL\H-board\DQN\tmp_Batch_10Decaps_Values.peb"

# decap Type
type1 = [22e-9 , 142e-12 , 25.2e-3]
type2 =  [47e-9 , 154e-12 , 21.4e-3]
type3 = [100e-9 , 222e-12 , 8.9e-3]
type_list = np.array([1,2,3])
num_type = len(type_list)
# define parameters
train_times = 2000
learning_rate =1e-3  # sher klein max 0.1 after each episode smaller alpha
discount_factor = 0.9 #gamma
epsilon = 0.9
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.999  # Exploration rate decay factor探索率衰减因子
fmin=1e5
fm=1e6
fmax=2e8
Zmin=4
Zmax=400


buffersize=int(1e6)
samplesize=1024
update_iteration=100
weight=0.5

##########################################################################################
#define function for class PCB
##########################################################################################

# peb.file generation
def gen_pebfiles(action):
    position = action[0]
    value = action[1]

    source = open(destination_file)
    tree = ET.parse(source)
    root = tree.getroot()

    k = position - 1

    if value == 0:
        print('No Action has been selected!')
        sys.exit()
    elif value == 1:
        root[0][0][4 * k].set('Value', 'true')
        root[0][0][4 * k + 1].set('Value', str(type1[0]))
        root[0][0][4 * k + 2].set('Value', str(type1[2]))
        root[0][0][4 * k + 3].set('Value', str(type1[1]))
    elif value == 2:
        root[0][0][4 * k].set('Value', 'true')
        root[0][0][4 * k + 1].set('Value', str(type2[0]))
        root[0][0][4 * k + 2].set('Value', str(type2[2]))
        root[0][0][4 * k + 3].set('Value', str(type2[1]))
    elif value == 3:
        root[0][0][4 * k].set('Value', 'true')
        root[0][0][4 * k + 1].set('Value', str(type3[0]))
        root[0][0][4 * k + 2].set('Value', str(type3[2]))
        root[0][0][4 * k + 3].set('Value', str(type3[1]))
    tree.write(destination_file)
    source.close()
# read current impedance and reward funktion
# global variables
num_t = 0  # the number of frequency points satisfying the target impedance at the steps t
num_t1 = 0  # the number of frequency points satisfying the target impedance at the steps t+1
def cal_reward(used_decap):
    # with open('/content/sample1'+ str(file_number+1) +'.csv', newline='') as csvfile:
    with open(user_name + r"\Desktop\PI_MARL\H-board\DQN\Design1.emc\PI-1/Power_GND/1-PIPinZ_IC1.csv", newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=';')
        next(reader, None)
        current_impedance_list = list(reader)

    # find the value in csv and delete other useless information,and turn the data into float
    frequency, impedance_values, num_rows=fh.get_freqency_and_impedance(current_impedance_list)
    R = 1  # final reward
    count = 0  # how many points satisfy the target impedance
    for i in range(num_rows):
        if impedance_values[i] <= fh.target_impedance(frequency[i],fmin,fmax,fm,Zmin,Zmax) :
            # datatype
            count += 1
    # aktualisieren the number of satisfied points
    global num_t, num_t1
    num_t = num_t1
    num_t1 = count
    empty_decap=state_einheit-used_decap
    if num_t1 == num_rows:  # if all the current-impedance < target-impedance
        return ((num_t1 - num_t) / (num_rows)) + (10 * (empty_decap/state_einheit)), True
    else:
        return (num_t1 - num_t) / (num_rows), False

##########################################################################################
#define class PCB
##########################################################################################

class PCBEnvironment:
    def __init__(self):
        self.actions = fh.possible_action(pos_list, type_list)

    def reset(self):
        global num_t, num_t1
        num_t = 0
        num_t1 = 0
        # initialize the current impedance and layout
        self.current_state  = np.zeros(( state_num ))
    def step(self, state, action,count):
        # initialize the current state after the selection of action
        next_state = fh.next_state_step(state,  action,state_einheit)
        self.current_state = next_state
        gen_pebfiles(action)
        fh.load_batch_file(destination_file)

        Reward_value, done = cal_reward(count)
        if (count==state_einheit or count==len(pos_list)) and done==False:
            Reward_value=-2

        return self.current_state, Reward_value, done

env = PCBEnvironment()

##########################################################################################
#define class DQN
##########################################################################################

class DeepQLearning:

    def __init__(self,env,learning_rate,discount_factor,epsilon,train_times,state_num,num_pos,num_type,epsilon_decay,buffersize,samplesize,update_iteration,weight):

        self.learning_rate=learning_rate
        self.env=env
        self.gamma=discount_factor
        self.epsilon=epsilon
        self.numberEpisodes=train_times
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay

        # state dimension
        self.stateDimension=state_num
        self.actionDimension=num_pos*num_type
        # this is the maximum size of the replay buffer
        self.replayBufferSize=buffersize
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize=samplesize

        # number of training episodes it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod=update_iteration

        # this is the counter for updating the target network
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork=0
        # replay buffer
        self.replayBuffer=deque(maxlen=self.replayBufferSize)

        # this is the main network
        # create network
        self.mainNetwork=self.createNetwork()

        # this is the target network
        # create network
        self.targetNetwork=self.createNetwork()

        # copy the initial weights to targetNetwork
        self.targetNetwork.set_weights(self.mainNetwork.get_weights())

        optimizer_nn = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.tou=weight

    # create a neural network
    def createNetwork(self):
        inp = Input((self.stateDimension))
        # Assemble shared layers
        x = keras.layers.Flatten()(inp)
        x = keras.layers.Dense(32, activation='relu', name="dense_1")(x)
        #x = tf.keras.layers.BatchNormalization()(x)  # Add batch normalization layer
        x = keras.layers.Dense(64, activation='relu', name="dense_2")(x)  # x = Dense(128, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu', name="dense_3")(x)
        outp = keras.layers.Dense(self.actionDimension, activation='linear', name="output_dense")(x)
        model = Model(inp, outp)
        optimizer_nn = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer_nn, loss='mse')
        return model

    def trainingEpisodes(self):
        result_list = []
        reward_list = []
        episode_num = []
        decaps_num = []
        reward_average=[]
        with open(file_path+'\Record'+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入CSV文件的表头，例如：Episode,State
            writer.writerow(['Episode', 'State'])
        # here we loop through the episodes
            for episode in range(self.numberEpisodes):
                self.env.reset()
                #check and stop loop
                if stop_loop:
                  episode=episode-1
                  break

                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                #print(" Double DQN episode ", episode+1, " begin time：", current_time)
                print(" DQN episode ", episode + 1, " begin time：", current_time)

                currentState = self.env.current_state
                count = 0
                fh.copy_batch_file(destination_file,source_file)
                reward_sum = 0
                while True:

                    if count== state_einheit or count==len(pos_list):
                        result_list.append(currentState)  # store the result
                        reward_list.append(reward)  # store the reward
                        episode_num.append(episode+1)
                        decaps_num.append(count)
                        reward_average.append(reward_sum / count)
                        print("not done and drop this result")
                        break
                    action_index=self.selectAction(currentState, episode)
                    action = self.env.actions[action_index]
                    count += 1
                    # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                    #next_state, reward, done = self.env.step(currentState, action,count)
                    next_state, reward, done = self.env.step(currentState, action, count)
                    if count==1 and done==True:
                        print("This is one outliers!!!DROP IT!!!")
                        break
                    print("step:", episode + 1, ".", count, "------Position:", action[0], "------Type:", action[1])
                    if (len(self.replayBuffer) >= self.batchReplayBufferSize):
                        tmp_next_state = np.expand_dims(next_state, axis=0)
                        tmp_currentState = np.expand_dims(currentState, axis=0)
                        if done:
                            target_value=reward
                            next_value=0
                        else:
                                #tmp_index = np.argmax(self.mainNetwork(tmp_next_state))
                                #next_value=(self.targetNetwork(tmp_next_state)[0][tmp_index])
                                #target_value = reward + self.gamma * (next_value)
                                next_value =  np.max(self.targetNetwork(tmp_next_state))
                                target_value = reward + self.gamma * next_value
                        current_value=(self.mainNetwork(tmp_currentState)[0][action_index]).numpy()
                        print("current value:",current_value,"------Reward:",reward,"------next value:",next_value,"------Target value:",target_value)
                    # add current state, action, reward, next state, and terminal flag to the replay buffer
                    self.replayBuffer.append((currentState,action_index,reward,next_state,done))
                    # train network
                    self.trainNetwork()

                    # set the current state for the next step
                    currentState=next_state
                    print("current layout:", currentState)
                    fh.copy_and_rename_files(episode, count)
                    reward_sum=reward_sum+reward
                    if done:
                            result_list.append(currentState)  # store the result
                            reward_list.append(reward)  # store the reward
                            episode_num.append(episode+1)
                            decaps_num.append(count)
                            reward_average.append(reward_sum/count)
                            break
                writer.writerow([episode + 1, currentState])
        return reward_average,result_list, reward_list, episode_num, decaps_num,episode

    def selectAction(self,state,index): # index - index of the current episode
        # first index episodes we select completely random actions to have enough exploration
        if index<1:
          while True:
            temp_action_index = np.random.randint(len(self.env.actions))  # select a random index of action
            temp_action = np.copy(self.env.actions[temp_action_index])  # find out the matrix of this action
            tmp_state_list=np.copy(state[:state_einheit])
            for i,value in enumerate(tmp_state_list):
                if value==0:
                    return temp_action_index
                if value==temp_action[0]:
                    break
        if index>200:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if np.random.uniform() < self.epsilon:
              while True:
                  temp_action_index = np.random.randint(len(self.env.actions))  # select a random index of action
                  temp_action = np.copy(self.env.actions[temp_action_index])  # find out the matrix of this action
                  tmp_state_list = np.copy(state[ :state_einheit])
                  for i, value in enumerate(tmp_state_list):
                      if value == 0:
                          return temp_action_index
                      if value == temp_action[0]:
                          break
        else:
              state = np.expand_dims(state, axis=0)
              q=self.mainNetwork(state).numpy()
              while True:
                temp_action_index = np.argmax(q)

                temp_action = self.env.actions[temp_action_index]  # find out the matrix of this action
                tmp_state_list = np.copy(state[0][ :state_einheit])
                for i, value in enumerate(tmp_state_list):
                    if value==0:
                        return temp_action_index
                    if value == temp_action[0]:
                        try:
                            q[0, temp_action_index] = -100
                        except:
                            q = fh.update_q(q, temp_action_index, -100)
                        break

    def trainNetwork(self):
        if (len(self.replayBuffer) >= self.batchReplayBufferSize):
            randomSampleBatch = random.sample(self.replayBuffer, self.batchReplayBufferSize)
            currentStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
            nextStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
            for index, tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index, :] = tupleS[0].flatten()  # .reshape(index)
                nextStateBatch[index, :] = tupleS[3].flatten()

            inputNetwork = currentStateBatch
            outputNetwork = np.zeros(shape=(self.batchReplayBufferSize, self.actionDimension))
            QnextStateTargetNetwork = self.targetNetwork(nextStateBatch)
            QcurrentStateMainNetwork = self.mainNetwork(currentStateBatch)

            for index, (currentState, action, reward, next_state, done) in enumerate(randomSampleBatch):
                y = reward + (1-done) * self.gamma * np.max(QnextStateTargetNetwork[index])

                # this actually does not matter since we do not use all the entries in the cost function
                outputNetwork[index] = QcurrentStateMainNetwork[index]
                # this is what matters
                outputNetwork[index, action] = y

            # here, we train the network
            self.mainNetwork.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize, verbose=0,
                                 epochs=100)
            self.counterUpdateTargetNetwork += 1
            if (self.counterUpdateTargetNetwork > (self.updateTargetNetworkPeriod - 1)):
                target_weights = [(self.tou * main_weight) + ((1 - self.tou) * target_weight)
                                    for main_weight, target_weight in
                                    zip(self.mainNetwork.get_weights(), self.targetNetwork.get_weights())]
                self.targetNetwork.set_weights(target_weights)
                print("Target network updated!")
                # reset the counter
                self.counterUpdateTargetNetwork = 0

##########################################################################################
#time, control and some list
##########################################################################################

stop_loop = False

def user_input_thread():
    global stop_loop
    input("Press 's' can stop the loop")
    stop_loop = True

input_thread = threading.Thread(target=user_input_thread)
input_thread.start()

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
timestamp1=current_time
dt1 = datetime.strptime(timestamp1, "%Y-%m-%d %H:%M:%S")


##########################################################################################
#the main part
##########################################################################################
folder_path= user_name + r"\Desktop\PI_MARL\H-board\DQN"
folder_name = "result_dqn"
file_path = os.path.join(folder_path, folder_name)

if not os.path.exists(file_path):
    os.makedirs(file_path)

print()
print(" train  start time:", current_time)
#double_dqn=False
# create an object
DQN=DeepQLearning(env,learning_rate,discount_factor,epsilon,train_times,state_num,num_pos,num_type,epsilon_decay,buffersize,samplesize,update_iteration,weight)
# run the learning process
#############################
#normal dqn
#############################
reward_average,result_list, reward_list, episode_num, decaps_num,episode = DQN.trainingEpisodes()
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(" normal DQN train end time:", current_time)
timestamp2=current_time
dt2 = datetime.strptime(timestamp2, "%Y-%m-%d %H:%M:%S")

time_diff = dt2 - dt1

# Extract the time difference components (days, hours, minutes, seconds)
days = time_diff.days
hours = time_diff.seconds // 3600
minutes = (time_diff.seconds // 60) % 60
seconds = time_diff.seconds % 60


# Print the time difference
print(f"Time difference: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

data1 = [[time_diff.days, time_diff.seconds // 3600, (time_diff.seconds // 60) % 60, time_diff.seconds % 60]]


current_date = now.strftime("%d_%m")

#  summarize the model
DQN.mainNetwork.summary()

data3 = [type1,type2,type3,[num_type,train_times,learning_rate,discount_factor,epsilon,epsilon_min,epsilon_decay,num_pos,fmin,fm,fmax,Zmin,Zmax]]

#data=[data1,data2,type1,type2,type3,train_times,learning_rate,discount_factor,epsilon,epsilon_min,epsilon_decay,fmin,fmax,fmittel,Zmin,Zmax,buffersize,samplesize,update_iteration,weight]

# Write the data to the CSV file
with open(file_path+r'\time_and_all_parameters_'+str(episode+1)+'_times_'+current_date+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Days", "Hours", "Minutes", "Seconds","for DQN"])  # Write the header
    writer.writerows(data1)  # Write the data row
    writer.writerow(["type1","type2","type3","num_type","train_times","learning_rate","discount_factor","epsilon","epsilon_min","epsilon_decay","fmin","fmax","fmittel","Zmin","Zmax","buffersize","samplesize"])
    writer.writerows(data3)     #_csv.Error: iterable expected, not int

current_date = now.strftime("%d_%m")

DQN.mainNetwork.save("trained_normal_DQN_model_"+str(episode+1)+"_times_"+current_date+".h5")
#Double_DQN.mainNetwork.save("trained_double_DQN_model_"+str(episode_double+1)+"_times_"+current_date+".h5")
print("Press s and Enter to stop and save")
input_thread.join()
##########################################################################################
#the evaluation
##########################################################################################

eva.plot_graph(reward_average,episode_num, decaps_num, reward_list, file_path, episode,current_date)
#eva.optimal_layout(decaps_num, result_list, file_path, episode, current_date,state_einheit,pos_list,type_list)
#eva.overdesigned_layout(decaps_num, result_list, file_path, episode, current_date,state_einheit,pos_list,type_list)
eva.result_csv(reward_average,episode_num,result_list,reward_list,decaps_num,file_path,episode)
