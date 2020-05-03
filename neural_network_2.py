import tensorflow as tf


import numpy as np
# import tensorflow as tf
import main
import sys
import os
import time
import csv
import argparse
import tensorflow.keras.backend as ksb
import yaml
import pandas as pd


class NeuralNetwork:
    """
    Neural network object for Q-learning.
    Provides an estimate of the Q-values given a current state.
    """

    def __init__(self, model_name, input_shape, num_actions, replay_memory, checkpoint_dir, training):
        """
        Args:
            n_inputs (int): Number of inputs
            num_actions (int): Number of possible actions in game environment
            replay_memory (obj): ReplayMemory object
        """

        self.model_name = model_name

        self.training = training
        self.count_states = tf.Variable(0.0, trainable=False) 
        self.count_episodes = tf.Variable(0.0, trainable=False)
        self.checkpoint_dir = checkpoint_dir
        self.replay_memory = replay_memory

        with open(checkpoint_dir + "nn_config.yaml", 'r') as yaml_file:
            self.nn_config = yaml.load(yaml_file, Loader=yaml.FullLoader)


        # Build model using layersizes from config
        layer_sizes = self.nn_config['layers']


        input_shape = [input_shape[0],]

        init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layer_sizes[0], activation='tanh', input_shape=input_shape, kernel_initializer=init))
        self.model.add(tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None))

        # regulariser = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        # , kernel_regularizer=regulariser
        for i in range(1, len(layer_sizes)):
            self.model.add(tf.keras.layers.Dense(layer_sizes[i], activation='tanh', kernel_initializer=init))
            self.model.add(tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None))

        self.model.add(tf.keras.layers.Dense(num_actions, activation='linear'))

        self.model.run_eagerly = False

        # Set training parameters
        huber_delta = self.nn_config['huber_delta']
        learn_rate  = self.nn_config['learning_rate']
        # self.loss = tf.keras.losses.Huber(delta=huber_delta)
        self.loss = tf.keras.losses.MeanAbsoluteError()
        # self.loss = tf.keras.losses.MeanSquaredError()

        # self.loss = self.custom_loss_2

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)


        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mae'])
        
        print(self.model.run_eagerly)

        # Create a callback that saves the model's weights
        # self.earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='min')
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, monitor='loss', mode='min', save_weights_only=False, verbose=1)


        # if model_name == "bvb_8":
        self.load_checkpoint()
            # exit(0)


    @tf.function
    def huber_loss_wrapper(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


    def load_checkpoint(self):
        """
        Load variables of graph from tf checkpoint.
        If checkpoint does not exist, variables are initialised
        from default parameters
        """
        try:
            self.model = tf.keras.models.load_model(self.checkpoint_dir, compile=False)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mae'])

        except Exception as e:
            print(e)
            print("unable to load model")
            # self.model.save(self.checkpoint_dir) 

        try:
            train_data_path = "./model_checkpoints/checkpoint_" + self.model_name + "/train_data.txt"
            df = pd.read_csv(train_data_path, sep=",", header=None)
            df.columns = ["count_states", "learning_rate", "mean_epsilon", "loss", "acc"]

            num_states = df['count_states'].values[-1]
            self.count_states = tf.Variable(num_states, trainable=False) 

        except Exception as e:
            print(e)
            print("unable to train_data, setting num states to zero")
            self.count_states = tf.Variable(0.0, trainable=False) 




            # continue


    def custom_loss(self, y_true, y_pred):
        loss = 0

        if y_pred.shape[0] == None:
            return loss

        for i in range(y_pred.shape[0]):
            dist = tf.gather(y_pred, [i], axis=0)
            y = y_true.numpy().squeeze()
            norm = tf.map_fn(tf.nn.l2_loss, dist-y_pred)

            d = norm.numpy()
            d[np.where(y != y[i])] = 0.0
            max_pos = tf.gather(norm, np.argmax(d))

            d = norm.numpy()
            d[np.where(y == y[i])] = np.inf
            min_neg = tf.gather(norm, np.argmin(d))

            clipped = tf.clip_by_value(max_pos - min_neg + tf.constant(1, dtype=tf.float32),
                                     1e-8, 1e1)


            loss += tf.clip_by_value(max_pos - min_neg + tf.constant(1, dtype=tf.float32),
                                     1e-8, 1e1)

        return tf.Variable(loss, trainable=False) 


    def custom_loss_2(self, y_true, y_pred):
        deltas = y_pred - y_true
        loss = tf.reduce_mean(deltas)

        print(loss.numpy())
        return loss

    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given
        states.

        Input is array pf states, processed in batch.

        Output is an array of Q-value-arrays, one Q-value for 
        each possible action. Shape [batch, num_actions]
        """

        # Calculate Q-values for the states

        # if self.model_name == "bvb_0":
        #     print("predict weights: ", self.model.get_weights()[0][1][0])

        Q_value_estimates = self.model.predict(states, batch_size=128)

        return Q_value_estimates


    def get_weight_grad(model, inputs, outputs):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)

        return output_grad


    def optimize(self, min_epochs=1.0, max_epochs=10,
        batch_size=16, loss_limit=0.0015, learning_rate=1e-3):
        """
        Optimize nn by sampling states and Q-values from the replay memory.
        """
        # self.learning_rate = learning_rate

        print("Optimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: {0:.1e}".format(learning_rate))
        print("\tLoss-limit: {0:.3f}".format(loss_limit))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        # self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

    	# for i in range(int(1e6)):
        # Randomly sample a batch of states and target Q-values
        # from the replay-memory. These are the Q-values that we
        # want the Neural Network to be able to estimate.
        states, q_values = self.replay_memory.random_batch_v6()
        print("before fit")
        print(self.model.get_weights()[0][0][0])

        self.optimizer.learning_rate = learning_rate
        self.model.fit(x=states, y=q_values, batch_size=batch_size, epochs=int(max_epochs), verbose=1, callbacks=[self.cp_callback])

        print("after fit")
        print(self.model.get_weights()[0][0][0])

        loss_val, acc = self.model.evaluate(states, q_values)

        loss_history = np.roll(loss_history, 1)
        loss_history[0] = loss_val
        loss_mean = np.mean(loss_history)

        prev_loss_val = loss_mean

        return loss_val, acc

    @tf.function
    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. Allows reloading of the counter
        with the tf checkpoint
        """
        return self.count_states.read_value()

    @tf.function
    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.count_episodes.read_value()

    @tf.function
    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.

        """
        self.count_states.assign_add(1.0)

    @tf.function
    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        self.count_episodes.assign_add(1.0)

    @tf.function
    def print_progress(self, msg):
        """
        Print progress on a single line and overwrite the line.
        Used during optimization.
        """

        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

