#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
# import tensorflow as tf

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
EXPLORE = 3000 # frames over which to anneal epsilon
FINAL_EPSILON = 0 # final value of epsilon
INITIAL_EPSILON = 0.05 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 256 # size of minibatch
LEARNING_RATE = 0.1

def buildmodel():
    model = Sequential()
    model.add(Dense(6, input_shape=[2]))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.compile(loss='mse',optimizer=Adam(lr=LEARNING_RATE))
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # store the previous observations in replay memory
    D = deque()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    s_t, r_0, terminal = game_state.frame_step(do_nothing)

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    if args['mode'] in ('Run', 'Cont'):
        if args['mode'] == 'Run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = FINAL_EPSILON
        model.load_weights("model.h5")
        model.compile(loss='mse',optimizer=Adam(lr=LEARNING_RATE))

    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = model.predict(s_t)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        s_t1, r_t, terminal = game_state.frame_step(a_t)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + np.max(Q_sa, axis=1)*(1-np.array(terminal))

            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t = t + 1
        # save progress every 10000 iterations
        if t % 1000 == 0:
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "| STATE", state, \
            "| EPSILON", epsilon, "| ACTION", action_index, "/ REWARD", r_t, \
            "| Q_MAX " , np.max(Q_sa), "| Loss ", loss)

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
