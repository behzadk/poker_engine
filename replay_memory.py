import numpy as np
import tensorflow as tf
import sys
import os
import time
import csv
import argparse
import pickle

class ReplayMemory:
    """
    Holds previous states of the game to be learned in batch
    """

    def __init__(self, size, state_shape, num_actions, discount_factor=0.97):
        """
        Args:
            size (int): Max number of states
            state_shape(arr): shape of state
            num_actions (int): Number of possible actions in game environment
            discount_factor (float): Discount factor for updating the Q-values

        """

        # Array of states
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array of Q-values. Contains an estimated q vale for EACH action
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Previous Q-values used to compare before and after update
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each state
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)


        # Estimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)


        self.size = size
        self.discount_factor = discount_factor

        # Number of states used
        self.num_used = 0

        # Threshold for choosing low and high estimator errors
        self.error_threshold = 0.1

        self.reward_clippig_min_max  = [-1.0, 1.0]


    def is_full(self):
        """ Returns bool indicating if replay memory is full """
        return self.size == self.num_used

    def used_fraction(self):
        """ Returns fraction of replay memory that is used"""
        return self.num_used / self.size

    def add(self, state, q_values, action, reward, end_episode):
        """
        Add an observed state of game environment with the estimated Q-values,
        actions and observed reward.
        """

        if not self.is_full():
            i = self.num_used
            self.num_used += 1

            # Store to replay memory
            self.states[i] = state
            self.q_values[i] = q_values
            self.actions[i] = action

            # if action == 0:
            #     print(reward)

            # Rewards clipped to prevent heavy weighting of outliers in training.
            # self.rewards[i] = np.clip(reward, self.reward_clippig_min_max[0], self.reward_clippig_min_max[1])

            self.rewards[i] = reward



    def update_all_q_values(self):
        """
        Updates all Q-values in the replay memory

        The estimates from the neural network are moved to the array for old.
        We then use the newly aquired data to improve the Q-value estimates (backwards sweep).

        """
        self.q_values_old[:] = self.q_values[:]
        idx = 0

        for i in reversed(range(self.num_used)):
            action = self.actions[i]
            reward = self.rewards[i]            

            action_value = reward #self.discount_factor * np.max(self.q_values[i])

            # Error of Q value estimation by the neural network
            # The difference between the actual value of the action taken and the estimated value
            # of that action
            self.estimation_errors[i] = abs(action_value - self.q_values[i, action])
            # print(self.estimation_errors[i])

            # print(action_value, self.q_values[i, action])

            # print(abs(action_value - self.q_values[i, action]))

            self.q_values[i, action] = action_value


    def prepare_sampling_prob(self, batch_size=128):
        """
        Binary split of replay memory based on the estimation error.

        Creates batch samples that are balanced evenly between Q-values 
        that the neural network is already doing well on, and those that it 
        has a poor estimation error of.

        If we train using Q-values on data with high estimation errors, it 
        will tend to forget the data it already knows.The balance aims to prevent the 
        neural net from overfitting or being unstable.
        """

        q_val_err = self.estimation_errors[0:self.num_used]

        rewards = self.rewards[0:self.num_used]

        extreme_rewards = [True if (x < 0.3 or x > 0.6) else False for x in rewards ]
        self.extreme_rewards_idxs = np.squeeze(np.where(extreme_rewards))

        # # Get index of errors that are "low"
        # low_errors = q_val_err < self.error_threshold
        # self.idx_err_low = np.squeeze(np.where(low_errors))

        # Get index of errors that are "high"
        self.idx_err_low = np.squeeze(np.where(np.logical_not(extreme_rewards)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.extreme_rewards_idxs) / self.num_used
        print(prob_err_hi)
        prob_err_hi = max(prob_err_hi, 0.5)
        prob_err_hi = 0.8


        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_high = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_low = batch_size - self.num_samples_err_high


    def random_batch(self):
        """
        Get a random batch of states and Q-values from replay memory. 

        The prepare_sampling_prob function is called before this generating
        the necessary object variables, so the batch size is already set.
        """

        # Randomly sample idxs with low error Q-values from the replay memory

        # print(np.shape(self.idx_err_low))
        # print(np.shape(self.idx_err_high))

        # print(self.num_samples_err_low)
        # print(self.num_samples_err_high)
        # print(self.num_used)
        # print(self.size)
        # exit()


        # idx_low = np.random.choice(self.idx_err_low, 
        #                             size=self.num_samples_err_low, 
        #                             replace=False)

        # # Randomly sample idxs with high error Q-values from replay memory
        # idx_high = np.random.choice(self.extreme_rewards_idxs, 
        #                             size=self.num_samples_err_high, 
        #                             replace=False)

        # # Indexes to sample
        # idx = np.concatenate((idx_low, idx_high))
        
        # # except:
        idx = np.random.choice(range(len(self.states)), 
                                size=self.num_samples_err_high + self.num_samples_err_low, 
                                replace=False)


        # Get the state and Q-value batch
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch


    def reset(self):
        """Reset the replay-memory by half so it is empty."""
        self.num_used = 0


    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")
        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))


