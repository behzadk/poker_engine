import numpy as np
import tensorflow as tf
import sys
import os
import time
import csv
import argparse
import main

from neural_network import NeuralNetwork
from rl_log import LogQValues, LogReward
from epsilon_greedy import EpsilonGreedy
from epsilon_greedy import LinearControlSignal
from neural_network_2 import NeuralNetwork
from replay_memory import ReplayMemory

from deuces import Card
import encode_state
import yaml

import pandas as pd

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
        self.checkpoint_dir = checkpoint_dir

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
                                            end_value=epsilon_testing,
                                            num_iterations=5e6,
                                            num_actions=self.num_actions,
                                            epsilon_testing=epsilon_testing)

        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=0.00001,
                                                             end_value=0.00001,
                                                             num_iterations=1e5)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.0,
                                                          end_value=0.0,
                                                          num_iterations=50000)

            # The maximum number of epochs to perform during optimization.
            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=1.0,
                                                          num_iterations=1e5)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=5e6)

        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM.
            self.replay_memory = ReplayMemory(size=16000, state_shape=self.state_shape,
                                              num_actions=self.num_actions, checkpoint_dir=checkpoint_dir)
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
        self.hand_rewards = []

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

        self.min_max_scaling = lambda a, b, min_x, max_x, x: a + ((x - min_x) * (b - a)) / (max_x - min_x)

        self.write_state_action = False
        self.output_path = "./output/player_actions/player_" + str(self.agent_name) + "_actions.csv"
        self.action_space = ['CALL', 'ALL_IN', 'CHECK', 'FOLD']

        with open(checkpoint_dir + "action_config.yaml", 'r') as yaml_file:
            self.action_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        raise_action_space = self.action_config['raise_actions']

        self.action_space.extend(raise_action_space)
        self.raise_idxs = list(range(4, len(raise_action_space) + 4))
        self.raise_multiples = self.action_config['raise_multiples']

        self.set_fold_q = self.action_config['set_fold_q']

    def get_replay_memory_size(self):
        return(self.replay_memory.num_used)

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
            q_values[3] = 0.0

            # Set call Q-value to zero
            q_values[0] = 0.0

            # Change raise space
            if table.current_bet + table.big_blind > hero_player.stack:
                # Set all raise Q values to zero
                q_values[4:] = 0.0
                valid_idxs = [1, 2]


            elif table.current_bet + table.big_blind*2 > hero_player.stack:
                # Set 2x raise + Q values to zero
                q_values[5:] = 0.0
                valid_idxs = [1, 2, 4]

            elif table.current_bet + table.big_blind*4 > hero_player.stack:
                # Set  4x raise + raise Q values to zero
                q_values[6:] = 0.0
                valid_idxs = [1, 2, 4, 5]

            elif table.current_bet + table.big_blind*8 > hero_player.stack:
                # Set  8x raise +  raise Q values to zero
                q_values[7:] = 0.0
                valid_idxs = [1, 2, 4, 5, 6]

            else:
                valid_idxs = [1, 2, 4, 5, 6, 7]



        ### Current bet above zero ###
        else:
            valid_idxs = [0, 1, 3, 4, 5, 6, 7]

            # Set check Q-value to zero
            q_values[2] = 0.0

            # Remove call if can only go all in or fold
            if table.current_bet > hero_player.stack:
                q_values[0] = 0.0
                q_values[4:] = 0.0
                valid_idxs = [1, 3]

            # Change raise space
            elif table.current_bet + table.big_blind > hero_player.stack:

                # Set all raise Q values to zero
                q_values[4:] = 0.0

                # Set call to 0
                q_values[0] = 0.0

                valid_idxs = [1, 3]


            elif table.current_bet + table.big_blind*2 > hero_player.stack:
                # Set 2x raise + Q values to zero
                q_values[5:] = 0.0
                valid_idxs = [0, 1, 3, 4]

            elif table.current_bet + table.big_blind*4 > hero_player.stack:
                # Set  4x raise + raise Q values to zero
                q_values[6:] = 0.0
                valid_idxs = [0, 1, 3, 4, 5]

            elif table.current_bet + table.big_blind*8 > hero_player.stack:
                # Set  8x raise +  raise Q values to zero
                q_values[7:] = 0.0
                valid_idxs = [0, 1, 3, 4, 5, 6]

            else:
                valid_idxs = [0, 1, 3, 4, 5, 6, 7]


        return q_values, valid_idxs


    def q_value_processing_v2(self, q_values, hero_player, table):
        self.action_type_space = ['CALL', 'ALL_IN', 'CHECK', 
        'FOLD', 'RAISE_1']


        if table.current_bet == 0:
            valid_idxs = [1, 2, 4]

            # Set call Q-value to zero
            q_values[0] = 0.0

            # Set fold Q-value to zero
            q_values[3] = 0.0

            # Change raise space
            if table.current_bet + table.big_blind > hero_player.stack:
                # Set all raise Q values to zero
                q_values[4] = 0.0
                valid_idxs = [1, 2]

        else:
            valid_idxs = [0, 1, 3, 4]

            if table.current_bet > hero_player.stack:
                q_values[0] = 0.0
                q_values[4] = 0.0
                valid_idxs = [1, 3]

            if table.current_bet + table.big_blind > hero_player.stack:
                q_values[0] = 0.0
                q_values[4] = 0.0

                valid_idxs = [1, 3]

        return q_values, valid_idxs


    def q_value_processing_v3(self, q_values, hero_player, table):
        # self.action_type_space = ['CALL', 'ALL_IN', 'CHECK', 
        # 'FOLD', 'RAISE_1', 'RAISE_1_5', 'RAISE_2', 'RAISE_2_5', 
        # 'RAISE_3', 'RAISE_3_5', 'RAISE_4', 'RAISE_4_5']

        current_bet = table.current_bet
        hero_stack = hero_player.stack
        big_blind = table.big_blind


        # Call, all in, check, fold

        if table.current_bet == 0:
            valid_base_idxs = [1, 2]
            valid_raise_idxs = [idx for idx, raise_mul in zip(self.raise_idxs, self.raise_multiples) if hero_stack >= (current_bet + big_blind * raise_mul) ]
            valid_idxs = valid_base_idxs + valid_raise_idxs

            q_values = [q if idx in valid_idxs else -1 for idx, q in enumerate(q_values)]

        else:
            valid_base_idxs = [0, 1, 3]

            valid_raise_idxs = [idx for idx, raise_mul in zip(self.raise_idxs, self.raise_multiples) if hero_stack >= (current_bet + big_blind * raise_mul) ]
            valid_idxs = valid_base_idxs + valid_raise_idxs
            q_values = [q if idx in valid_idxs else -1 for idx, q in enumerate(q_values)]


        return q_values, valid_idxs


    def get_action(self, hero_player, table, state):
        """
        Called by the game, requesting a response from the agent.
        """

        q_values = self.model.get_q_values(states=state)[0]


        if self.set_fold_q:
            norm_fold_value = self.min_max_scaling(-1, 1, 0, 2, hero_player.stack / hero_player.prev_stack)
            q_values[3] = norm_fold_value

        # norm_reward_hill = lambda x, k: x**1 / (k**1 + x**1 + 1e-12)
        # q_values[0] = norm_reward_hill(q_values[0], 1.0)


        processed_q_values, valid_idxs = self.q_value_processing_v3(q_values, hero_player, table)
        # valid_idxs = [0, 1]
        # min_max_scaling = lambda a, b, min_x, max_x, x: a + ((x - min_x) * (b - a)) / (max_x - min_x)

        # q_values[3] = hero_player.stack / hero_player.prev_stack
        count_states = self.model.get_count_states()

        # Determine the action that the agent must take in the game-environment.
        # The epsilon is just used for printing further below.
        action, epsilon = self.epsilon_greedy.get_action(q_values=processed_q_values,
                                                         iteration=count_states,
                                                         training=self.training,
                                                         valid_idxs=valid_idxs)
    

        # Card.print_pretty_cards(table.board)
        # print(self.agent_name, Card.print_pretty_cards(hero_player.hand))
        # print("Q-values: ",  self.action_space[action])
        # print("")
        # if self.agent_name == "model_6":
        #     Card.print_pretty_cards(hero_player.hand)
        #     Card.print_pretty_cards(table.board)
        #     print("Q-values: ", q_values)
        #     print("current stack: ", hero_player.stack)
        #     print("current bet: ", table.current_bet)
        #     print("Q-values: ",  self.action_space[action])
        #     print("")

        # if self.write_state_action:
        #     self.generate_state_action_data(state, q_values, table, hero_player)

        # else:
        self.episode_states.append(state)
        self.episode_q_values.append(q_values)
        self.episode_actions.append(action)
        self.episode_epsilons.append(epsilon)

        return action


    def update_end_hand_reward(self, end_hand_reward):
        total_investment = np.sum(self.hand_rewards) / len(self.hand_rewards)

        if total_investment == 0.0:
            proportional_rewards = [0.0 for x in self.hand_rewards]
            proportional_rewards[0] = end_hand_reward / len(self.hand_rewards)

        else:
           proportional_rewards = [(end_hand_reward *  x) / total_investment for x in self.hand_rewards]

        # print(end_hand_reward)
        # print(self.hand_rewards)
        # print(proportional_rewards)
        # print("")
        # updated_hand_rewards = [x + end_hand_reward for x in self.hand_rewards]
        
        for x in proportional_rewards:
            self.episode_rewards.append(x)

        self.hand_rewards = []

    def update_end_episode_reward(self, end_episode_reward):
        self.episode_rewards = [x + end_episode_reward for x in self.episode_rewards]

    def update_replay_memory(self):
        """
        Needs to be called at the end of an episode, then we update
        """
        win_rate = 0

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.increase_count_episodes()
        
        is_full = False
        # episode_rewards = [0 for i in range(len(self.episode_states))]
        # episode_rewards[-1] = end_hand_reward

        # If we want to train the Neural Network to better estimate Q-values.
        if self.training:
            for x in range(len(self.episode_states)):
                end_episode = False

                # Add the state of the game-environment to the replay-memory.
                self.replay_memory.add(state=self.episode_states[x],
                                       q_values=self.episode_q_values[x],
                                       action=self.episode_actions[x],
                                       reward=self.episode_rewards[x],
                                       end_episode=end_episode)

                self.model.increase_count_states()


                # print(self.episode_rewards)


            # How much of the replay-memory should be used.
            count_states = self.model.get_count_states()
            use_fraction = self.replay_fraction.get_value(iteration=count_states)

            print(self.replay_memory.used_fraction())
            # print(self.replay_memory.used_fraction())
            # print("")
            

            # When the replay-memory is sufficiently full.

            if self.replay_memory.is_full() \
                or self.replay_memory.used_fraction() > use_fraction:
                
                is_full = True


                print("fraction full")
                # Update all Q-values in the replay-memory through a backwards-sweep.
                self.replay_memory.update_all_q_values()
                print(np.around(self.replay_memory.estimation_errors,decimals=2))
                # print(self.replay_memory.estimation_errors)
                # exit()

                # Log statistics for the Q-values to file.

                # Get the control parameters for optimization of the Neural Network.
                # These are changed linearly depending on the state-counter.
                learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                # Perform an optimization run on the Neural Network so as to
                # improve the estimates for the Q-values.
                # This will sample random batches from the replay-memory.
                loss_mean, acc = self.model.optimize(learning_rate=learning_rate, loss_limit=loss_limit, max_epochs=max_epochs)

                mean_epsilon = np.mean(self.episode_epsilons)

                print()
                msg = "{0:.1f}, {1:.4f}, {2:.3f}, {3}, {4:.4f}\n".format(count_states, learning_rate, mean_epsilon, loss_mean, acc)
                with open(file=self.checkpoint_dir + "train_data.txt", mode='a', buffering=1) as file:
                    file.write(msg)

                # Reset the replay-memory. This throws away all the data we have
                # just gathered, so we will have to fill the replay-memory again.
                self.replay_memory.reset()

        self.reset_episode_rewards()

        return is_full



    def train_from_history_csv(self, learning_rate):
        history_path = self.checkpoint_dir + "state_reward.csv"
        
        print("Reading csv...")
        df = pd.read_csv(history_path)

        df_shape = np.shape(df)

        # make column names
        col_names = ["state_" +  str(i) for i in range(df_shape[1] - 2)]
        col_names.append("action")
        col_names.append("reward")
        df.columns = col_names

        states =  df.loc[:, df.columns != 'action']
        states =  states.loc[:, states.columns != 'reward'].values

        actions = df['action'].values
        rewards = df['reward'].values

        self.replay_memory = ReplayMemory(size=len(actions), state_shape=self.state_shape,
                                              num_actions=self.num_actions, checkpoint_dir=self.checkpoint_dir)

        self.replay_memory.save_state_reward = False
        # Create the Neural Network used for estimating Q-values.
        self.model = NeuralNetwork(model_name=self.agent_name, input_shape=self.state_shape, num_actions=self.num_actions, 
            checkpoint_dir=self.checkpoint_dir, replay_memory=self.replay_memory, training=self.training)

        states = np.array(states)

        print("Generating Q values...")
        q_values = self.model.get_q_values(states=states)

        print("Adding states to replay memory")
        # Get Q value predictions and add to replay memory
        for idx in range(len(actions)):

            # Add the state of the game-environment to the replay-memory.
            self.replay_memory.add(state=states[idx],
                                   q_values=q_values[idx],
                                   action=actions[idx],
                                   reward=rewards[idx],
                                   end_episode=False)

        print("Updating Q values")
        self.replay_memory.update_all_q_values()

        # Get the control parameters for optimization of the Neural Network.
        # These are changed linearly depending on the state-counter.
        # learning_rate = self.learning_rate_control.get_value(iteration=count_states)
        # loss_limit = self.loss_limit_control.get_value(iteration=count_states)
        # max_epochs = self.max_epochs_control.get_value(iteration=count_states)

        # Perform an optimization run on the Neural Network so as to
        # improve the estimates for the Q-values.
        # This will sample random batches from the replay-memory.
        print("Running optimization")
        loss_mean, acc = self.model.optimize(learning_rate=learning_rate, loss_limit=0, max_epochs=1)



