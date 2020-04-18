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

    def __init__(self, size, state_shape, num_actions, discount_factor=0.99):
        """
        Args:
            size (int): Max number of states
            state_shape(arr): shape of state
            num_actions (int): Number of possible actions in game environment
            discount_factor (float): Discount factor for updating the Q-values

        """

        # Array of states
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.float)

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

        # print("num_used: ", self.num_used)

        for i in reversed(range(self.num_used - 1)):
            action = self.actions[i]
            reward = self.rewards[i]            
            # print("q_values: ", self.q_values[i+1])

            # print(i)
            # print("reward: ", reward)
            action_value = reward 
            # + self.discount_factor * np.max(self.q_values[i + 1])
            # print("action value: ", action_value)
            # print("max Q: ", np.max(self.q_values[i + 1]))
            # print("")
            # Error of Q value estimation by the neural network
            # The difference between the actual value of the action taken and the estimated value
            # of that action
            self.estimation_errors[i] = abs(action_value - self.q_values[i, action])


            # print(self.estimation_errors[i])

            # print("av: ", action_value)
            # print("reward: ", reward)

            # print(action)
            # print(self.states[i])
            # print("")

            # print(self.estimation_errors[i])

            # print(action_value, self.q_values[i, action])

            # print(abs(action_value - self.q_values[i, action]))

            self.q_values[i, action] = action_value


    # def prepare_sampling_prob(self, batch_size=128):
    #     """
    #     Binary split of replay memory based on the estimation error.

    #     Creates batch samples that are balanced evenly between Q-values 
    #     that the neural network is already doing well on, and those that it 
    #     has a poor estimation error of.

    #     If we train using Q-values on data with high estimation errors, it 
    #     will tend to forget the data it already knows.The balance aims to prevent the 
    #     neural net from overfitting or being unstable.
    #     """


    #     unique_actions, counts = np.unique(self.actions, return_counts=True)
    #     batch_props = int(batch_size / len(unique_actions))
    #     self.num_samples = batch_size

    def prepare_sampling_prob(self, batch_size=128):
        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err<self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))
        # self.idx_err_lo = [x for x in self.idx_err_lo if self.actions[x] == 3]

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))
        # self.idx_err_hi = [x for x in self.idx_err_hi if self.actions[x] == 3]


        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        # prob_err_hi = 

        #max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

        self.num_samples = self.num_samples_err_lo + self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from replay memory. 

        The prepare_sampling_prob function is called before this generating
        the necessary object variables, so the batch size is already set.
        """


        unique_actions, counts = np.unique(self.actions, return_counts=True)
        running_num_samples = self.num_samples

        all_sampled_idxs = []

        count = 0
        for action_val, action_count in zip(unique_actions, counts):
            if action_val != 3:
                continue

            batch_prop = int(running_num_samples / (len(unique_actions) - count))
            sub_set = np.argwhere(self.actions == action_val).reshape(1, -1)[0]
            sample_x = np.min( [batch_prop, action_count] )

            sampled_action_indexes = np.random.choice(sub_set, size=sample_x, replace=False)

            all_sampled_idxs.extend(sampled_action_indexes)

            running_num_samples = running_num_samples - sample_x
            count += 1


        np.random.shuffle(all_sampled_idxs)

        states_batch = self.states[all_sampled_idxs]
        q_values_batch = self.q_values[all_sampled_idxs]

        # # Get the state and Q-value batch
        # states_batch = self.states[idx]
        # q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch


    def random_batch_v2(self):
        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch


    def random_batch_v3(self):
        unique_actions, counts = np.unique(self.actions, return_counts=True)
        running_num_samples = self.num_samples

        all_sampled_idxs = []

        count = 0
        for action_val, action_count in zip(unique_actions, counts):
            if action_val not in [1]:
                continue

            batch_prop = int(running_num_samples / (len(unique_actions) - count))
            sub_set = np.argwhere(self.actions == action_val).reshape(1, -1)[0]
            # sample_x = np.min( [batch_prop, action_count] )


            err = self.estimation_errors[sub_set]

            sample_x = np.min( [batch_prop, action_count] )
            sample_x = int(np.floor(sample_x/2))


            # Create an index of the estimation errors that are low.
            idx = err < self.error_threshold


            idx_err_lo = np.squeeze(np.where(idx))
            idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))
            
            sample_low = int(np.min([sample_x, len(idx_err_lo)]))


            idx_lo = np.random.choice(idx_err_lo,
                                      size=sample_low,
                                      replace=False)
            idx_lo = sub_set[idx_lo]


            sample_high = int(np.min([sample_x, len(idx_err_hi)]))

            # Random index of states and Q-values in the replay-memory.
            # These have HIGH estimation errors for the Q-values.
            idx_hi = np.random.choice(idx_err_hi,
                                      size=sample_high,
                                      replace=False)

            idx_hi = sub_set[idx_hi]

            all_sampled_idxs.extend(idx_hi)
            all_sampled_idxs.extend(idx_lo)

            running_num_samples = running_num_samples - sample_low + sample_high
            count += 1

        np.random.shuffle(all_sampled_idxs)
        states_batch = self.states[all_sampled_idxs]
        q_values_batch = self.q_values[all_sampled_idxs]
        actions_batch = self.actions[all_sampled_idxs]

        return states_batch, q_values_batch


    def random_batch_v4(self):
        unique_actions, counts = np.unique(self.actions, return_counts=True)
        running_num_samples = self.num_samples

        all_sampled_idxs = []

        count = 0
        for action_val, action_count in zip(unique_actions, counts):
            use_high_low_sampling = True

            # if action_val not in [1]:
            #     continue

            if action_val in [0, 1, 2, 3, 4]:
                use_high_low_sampling = False

            if action_val == 3:
                continue


            batch_prop = int(running_num_samples / (len(unique_actions) - count))
            sub_set = np.argwhere(self.actions == action_val).reshape(1, -1)[0]
            # sample_x = np.min( [batch_prop, action_count] )


            sample_x = np.min( [batch_prop, action_count] )

            if use_high_low_sampling:
                err = self.estimation_errors[sub_set]
                sample_x = int(np.floor(sample_x/2))

                idx = err < self.error_threshold
                idx_err_lo = np.squeeze(np.where(idx))
                idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))
                
                sample_low = int(np.min([sample_x, len(idx_err_lo)]))
                idx_lo = np.random.choice(idx_err_lo,
                                          size=sample_low,
                                          replace=False)
                idx_lo = sub_set[idx_lo]


                sample_high = int(np.min([sample_x, len(idx_err_hi)]))

                # Random index of states and Q-values in the replay-memory.
                # These have HIGH estimation errors for the Q-values.
                idx_hi = np.random.choice(idx_err_hi,
                                          size=sample_high,
                                          replace=False)

                idx_hi = sub_set[idx_hi]

                all_sampled_idxs.extend(idx_hi)
                all_sampled_idxs.extend(idx_lo)
                running_num_samples = running_num_samples - sample_low + sample_high


            else:
                idx = np.random.choice(sub_set, size=sample_x, replace=False)

                running_num_samples = running_num_samples - sample_x
                all_sampled_idxs.extend(idx)

            count += 1

        np.random.shuffle(all_sampled_idxs)
        states_batch = self.states[all_sampled_idxs]
        q_values_batch = self.q_values[all_sampled_idxs]
        actions_batch = self.actions[all_sampled_idxs]

        return states_batch, q_values_batch

    def random_batch_v5(self):
        all_indexes = np.argwhere(self.actions != 3).reshape(1, -1)[0]

        chosen_idxs = np.random.choice(all_indexes, size=self.num_samples)

        states_batch = self.states[chosen_idxs]
        q_values_batch = self.q_values[chosen_idxs]
        actions_batch = self.actions[chosen_idxs]

        return states_batch, q_values_batch

    def random_batch_v6(self):
        # all_indexes = np.argwhere(self.actions != 3).reshape(1, -1)[0]
        all_indexes = list(range(len(self.actions)))
        np.random.shuffle(all_indexes)

        # all_indexes = all_indexes[0:10]


        # states = [x for idx, x in enumerate(self.states) if idx in all_indexes]
        # q_values = [x for idx, x in enumerate(self.q_values) if idx in all_indexes]

        states = self.states[all_indexes]
        q_values = self.q_values[all_indexes]

        # print(np.shape(self.states))
        # print(states)
        # print(q_values)
        # exit()
        return states, q_values


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


