import numpy as np
import tensorflow as tf
import sys
import os
import time
import csv
import argparse

from neural_network import NeuralNetwork
from rl_log import LogQValues, LogReward
from epsilon_greedy import EpsilonGreedy
from epsilon_greedy import LinearControlSignal
from neural_network import NeuralNetwork
from replay_memory import ReplayMemory

from deuces import Card
class Agent:
    """
    Agent class interacts with the game evironment and creates
    instances of replay memory and the neural network
    """
    def __init__(self, agent_name, action_names, training, epsilon_testing,
        state_shape, checkpoint_dir, render=False, use_logging=True):
        """
        Create agent object instance. Will initialise the replay memory
        and neural network
        
        Args:
            agent_name (str): Name of agent
            training (bool): Whether the agent is training the neural network (True)
                            or playing in test model (False)
            render (bool): Wjether to render the game (redundant)
            use_logging (bool): Whether to log to text files during training

        """
        self.agent_name = agent_name

        # The number of possible actions that the agent may take in every step.
        self.num_actions = len(action_names)

        # Whether we are training (True) or testing (False).
        self.training = training

        # Whether to render each image-frame of the game-environment to screen.
        self.render = render

        # Whether to use logging during training.
        self.use_logging = use_logging

        # Set shape of state that will be input
        self.state_shape = state_shape

        if self.use_logging and self.training:
            # Used for logging Q-values and rewards during training.
            self.log_q_values = LogQValues()
            self.log_reward = LogReward()
        else:
            self.log_q_values = None
            self.log_reward = None


        # List of string-names for the actions in the game-environment.
        self.action_names = action_names

        # Initialise epsilon greedy
        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=800000,
                                            num_actions=self.num_actions,
                                            epsilon_testing=epsilon_testing)

        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=0.001,
                                                             end_value=0.001,
                                                             num_iterations=800000)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.03,
                                                          end_value=0.01,
                                                          num_iterations=800000)

            # The maximum number of epochs to perform during optimization.
            self.max_epochs_control = LinearControlSignal(start_value=10.0,
                                                          end_value=30.0,
                                                          num_iterations=800000)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.8,
                                                       end_value=1.0,
                                                       num_iterations=800000)

        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM.
            self.replay_memory = ReplayMemory(size=1000, state_shape=self.state_shape,
                                              num_actions=self.num_actions)
        else:
            self.replay_memory = None

        # Create the Neural Network used for estimating Q-values.
        self.model = NeuralNetwork(model_name=agent_name, input_shape=self.state_shape, num_actions=self.num_actions, 
            checkpoint_dir=checkpoint_dir, replay_memory=self.replay_memory, training=self.training)

        # Record episode states. In the case of poker,
        # a hand constitutes an episode.
        self.episode_states = []
        self.episode_q_values = []
        self.episode_actions = []
        self.episode_epsilons = []

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

    def reset_episode_rewards(self):
        """Reset the log of episode-rewards."""
        self.episode_states = []
        self.episode_q_values = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_epsilons = []

    def get_action_name(self, action):
        """Return the name of an action."""
        return self.action_names[action]

    def q_value_processing(self, q_values, hero_player, table):
        if table.current_bet == 0:
            valid_idxs = [1, 2, 4, 5, 6, 7]

            # Set fold Q-value to zero
            q_values[0][3] = 0.0

            # Set call Q-value to zero
            q_values[0][0] = 0.0

            # Change raise space
            if table.current_bet + table.big_blind > hero_player.stack:
                # Set all raise Q values to zero
                q_values[0][4:] = 0.0
                valid_idxs = [1, 2]


            elif table.current_bet + table.big_blind*2 > hero_player.stack:
                # Set 2x raise + Q values to zero
                q_values[0][5:] = 0.0
                valid_idxs = [1, 2, 4]

            elif table.current_bet + table.big_blind*4 > hero_player.stack:
                # Set  4x raise + raise Q values to zero
                q_values[0][6:] = 0.0
                valid_idxs = [1, 2, 4, 5]

            elif table.current_bet + table.big_blind*8 > hero_player.stack:
                # Set  8x raise +  raise Q values to zero
                q_values[0][7:] = 0.0
                valid_idxs = [1, 2, 4, 5, 6]

            else:
                valid_idxs = [1, 2, 4, 5, 6, 7]



        ### Current bet above zero ###
        else:
            valid_idxs = [0, 1, 3, 4, 5, 6, 7]

            # Set check Q-value to zero
            q_values[0][2] = 0.0

            # Remove call if can only go all in or fold
            if table.current_bet > hero_player.stack:
                q_values[0][0] = 0.0
                q_values[0][4:] = 0.0
                valid_idxs = [1, 3]

            # Change raise space
            elif table.current_bet + table.big_blind > hero_player.stack:
                # Set all raise Q values to zero
                q_values[0][4:] = 0.0
                valid_idxs = [0, 1, 3]


            elif table.current_bet + table.big_blind*2 > hero_player.stack:
                # Set 2x raise + Q values to zero
                q_values[0][5:] = 0.0
                valid_idxs = [0, 1, 3, 4]

            elif table.current_bet + table.big_blind*4 > hero_player.stack:
                # Set  4x raise + raise Q values to zero
                q_values[0][6:] = 0.0
                valid_idxs = [0, 1, 3, 4, 5]

            elif table.current_bet + table.big_blind*8 > hero_player.stack:
                # Set  8x raise +  raise Q values to zero
                q_values[0][7:] = 0.0
                valid_idxs = [0, 1, 3, 4, 5, 6]

            else:
                valid_idxs = [0, 1, 3, 4, 5, 6, 7]


        return q_values, valid_idxs


    def get_action(self, hero_player, table, state, fold_state):
        """
        Called by the game, requesting a response from the agent.

        """
        q_values = self.model.get_q_values(states=state)[0]
        fold_q_values = self.model.get_q_values(states=fold_state)[0][0]

        q_values[0] = fold_q_values


            # exit()

        # q_values, valid_idxs = self.q_value_processing(q_values, hero_player, table)
        valid_idxs = [0, 1]

        count_states = self.model.get_count_states()

        # Determine the action that the agent must take in the game-environment.
        # The epsilon is just used for printing further below.
        action, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                         iteration=count_states,
                                                         training=self.training,
                                                         valid_idxs=valid_idxs)
        if self.agent_name == "bvb_0":
            print(state[0][1:3][0] * 13, state[0][1:3][1] * 13)
            print(q_values)
            print(action)
            print("")

        count_states = self.model.increase_count_states()


        if action == 0:
            self.episode_states.append(fold_state)

        else:
            self.episode_states.append(state)

        self.episode_q_values.append(q_values)
        self.episode_actions.append(action)
        self.episode_epsilons.append(epsilon)


        return action

    def update_replay_memory(self, end_hand_reward):
        """
        Needs to be called at the end of an episode, then we update
        """
        count_states = self.model.get_count_states()

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.get_count_episodes()
        count_episodes = self.model.increase_count_episodes()


        # If we want to train the Neural Network to better estimate Q-values.
        if self.training:
            for x in range(len(self.episode_states)):
                end_episode = False
                # if x == len(self.episode_states):
                #     end_episode = True
                # else:
                #     end_episode = False

                count_states = self.model.increase_count_states()


                # Add the state of the game-environment to the replay-memory.
                self.replay_memory.add(state=self.episode_states[x],
                                       q_values=self.episode_q_values[x],
                                       action=self.episode_actions[x],
                                       reward=end_hand_reward,
                                       end_episode=end_episode)

            # How much of the replay-memory should be used.
            count_states = self.model.get_count_states()
            use_fraction = self.replay_fraction.get_value(iteration=count_states)

            # print(use_fraction)
            # print(self.replay_memory.used_fraction())
            # print("")
            

            # When the replay-memory is sufficiently full.
            if self.replay_memory.is_full() \
                or self.replay_memory.used_fraction() > use_fraction:
                
                # Update all Q-values in the replay-memory through a backwards-sweep.
                self.replay_memory.update_all_q_values()

                # Log statistics for the Q-values to file.
                if self.use_logging:
                    self.log_q_values.write(count_episodes=count_episodes,
                                            count_states=count_states,
                                            q_values=self.replay_memory.q_values)

                # Get the control parameters for optimization of the Neural Network.
                # These are changed linearly depending on the state-counter.
                learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                # Perform an optimization run on the Neural Network so as to
                # improve the estimates for the Q-values.
                # This will sample random batches from the replay-memory.
                self.model.optimize(learning_rate=learning_rate,
                                    loss_limit=loss_limit,
                                    max_epochs=max_epochs)

                # Save a checkpoint of the Neural Network so we can reload it.
                self.model.save_checkpoint(count_states)
                

                # Reset the replay-memory. This throws away all the data we have
                # just gathered, so we will have to fill the replay-memory again.
                self.replay_memory.reset()

                # Print reward to screen.
                if len(self.episode_epsilons) > 0:
                    episode_mean  = self.episode_epsilons[-1] / len(self.episode_epsilons)
                    msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
                    print(msg.format(count_episodes, count_states, self.episode_epsilons[-1],
                                     end_hand_reward, episode_mean))
                    print(self.replay_fraction.get_value(iteration=count_states))
            # if self.use_logging:
            #     self.log_reward.write(count_episodes=count_episodes,
            #                           count_states=count_states,
            #                           reward_episode=end_hand_reward,
            #                           reward_mean=end_hand_reward)


            count_states = self.model.get_count_states()

            # Counter for the number of episodes we have processed.
            count_episodes = self.model.get_count_episodes()


        self.reset_episode_rewards()

