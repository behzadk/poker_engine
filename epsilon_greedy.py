import numpy as np
import tensorflow as tf
import sys
import os
import time
import csv
import argparse

class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    
    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Args:
            start_value (float): Initial epsilon value
            end_value (float): Final epsilon value
            num_iterations (int): Number of iterations that linearly separate the start and end epsilon
            repeat (bool): Whether to repeat the linear decrease once the end value is reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """
        Get the value of the control signal for the given iteration.
        """

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value


class EpsilonGreedy:
    """
    Epsilon greedy policy has a probability of taking a random action, or the action
    with the highest Q-value.

    When epsilon is 1.0, all actions are random
    When epsilon is 0.0, all actions are from the highest Q-value

    Testing, epsilon should be 0.01 to 0.05
    """

    def __init__(self, num_actions, epsilon_testing=0.05,
        num_iterations=1e6, start_value=1.0, end_value=0.1, repeat=False):
        """
        Args:
            num_actions (int): Number of possible actions in game environment
            epsilon_testing (float): Epsilon value when testing
            num_iterations (int): Number of iterations that linearly separate the start and end epsilon
            start_value (float): Initial epsilon value
            end_value (float): Final epsilon value
            repeat (bool): Whether to repeat the linear decrease once the end value is reached.
        """

        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,  start_value=start_value, 
                                                end_value=end_value, repeat=repeat)


    def get_epsilon(self, iteration, training):
        """
        Args:
            iteration (int): Counter for number of states we have iterated.
            training (bool): Boolean for if training.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)

        else:
            epsilon = self.epsilon_testing

        return epsilon


    def get_action(self, q_values, iteration, training):
        """
        Epsilon greedy policy to choose an action

        Args:
            q_values (arr): Q-values estimated by the neural network
            iteration (int): Counter for number of states we have iterated.
            training (bool): Boolean for if training.
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # Probability of choosing random action
        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=self.num_actions)

        else:
            action = np.argmax(q_values)

        return action, epsilon
