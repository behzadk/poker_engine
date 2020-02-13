import numpy as np
# import tensorflow as tf
import sys
import os
import time
import csv
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class NeuralNetwork:
    """
    Neural network object for Q-learning.
    Provides an estimate of the Q-values given a current state.
    """

    def __init__(self, input_shape, num_actions, replay_memory, checkpoint_dir):
        """
        Args:
            n_inputs (int): Number of inputs
            num_actions (int): Number of possible actions in game environment
            replay_memory (obj): ReplayMemory object
        """

        # Optional parameters
        h_layer_units = input_shape[1] #- input_shape[1]/2


        self.replay_memory = replay_memory
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

        # Placeholder variable for input state 
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='x')

        # Placeholder for the learning rate.
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Placeholder variable for inputting the target Q-value that we want to estimate
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, num_actions],
                                           name='q_values_new')

        # From Hvass:
        # This is a hack that allows us to save/load the counter for
        # the number of states processed in the game-environment.
        # We will keep it as a variable in the TensorFlow-graph
        # even though it will not actually be used by TensorFlow.
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')


        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')


        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)


        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        
        # Neural network architechture

        # Initialise weights to be close to zero.
        init = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)

        activation = tf.nn.relu

        net = self.x

        # First fully-connected (aka. dense) layer.
        net = tf.layers.dense(inputs=net, name='layer_fc1', units=num_actions,
                              kernel_initializer=init, activation=activation)

        # # First fully-connected (aka. dense) layer.
        # net = tf.layers.dense(inputs=net, name='layer_fc2', units=num_actions,
        #                       kernel_initializer=init, activation=activation)

        
        # The output of the Neural Network is the estimated Q-values
        # for each possible action in the game-environment.
        # RMSPropOptimizer is adaptive stochastic gradient descent, Root mean squared (RMS).
        self.q_values = net

        # L2-loss is the difference between
        squared_error = tf.square(self.q_values - self.q_values_new)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        self.loss = tf.reduce_mean(sum_squared_error)

        # Optimizer to minimise the loss function.
        # learning rate is a placeholder defined earlier, because this needs
        # to change dynamically as the optimization progresses
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # For saving checkpoints
        self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        self.session = tf.Session()

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        self.load_checkpoint(checkpoint_dir)


    def close(self):
        self.session.close()

    def load_checkpoint(self, checkpoint_dir):
        """
        Load variables of graph from tf checkpoint.
        If checkpoint does not exist, variables are initialised
        from default parameters
        """

        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            self.session.run(tf.global_variables_initializer())


    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        print("Saved checkpoint.")


    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given
        states.

        Input is array pf states, processed in batch.

        Output is an array of Q-value-arrays, one Q-value for 
        each possible action. Shape [batch, num_actions]
        """

        # Create feed dict for inputing states to nn
        feed_dict = {self.x: states}


        # Calculate Q-values for the states
        Q_value_estimates = self.session.run(self.q_values, feed_dict=feed_dict)

        return Q_value_estimates


    def optimize(self, min_epochs=1.0, max_epochs=10, 
        batch_size=128, loss_limit=0.0015, learning_rate=1e-3):
        """
        Optimize nn by sampling states and Q-values from the replay memory.
        """

        print("Optimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: {0:.1e}".format(learning_rate))
        print("\tLoss-limit: {0:.3f}".format(loss_limit))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Number of optimization iterations corresponding to one epoch.
        iterations_per_epoch = self.replay_memory.num_used / batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)

        # Maximum number of iterations to perform.
        max_iterations = int(iterations_per_epoch * max_epochs)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)


        for i in range(max_iterations):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            state_batch, q_values_batch = self.replay_memory.random_batch()

            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.
            feed_dict = {self.x: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}

            np.shape(feed_dict)

            # Perform one optimization step and get the loss-value.
            loss_val, _ = self.session.run([self.loss, self.optimizer],
                                           feed_dict=feed_dict)

            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            pct_epoch = i / iterations_per_epoch
            msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}"
            msg = msg.format(i, pct_epoch, loss_val, loss_mean)
            self.print_progress(msg)

            # Stop the optimization if we have performed the required number
            # of iterations and the loss-value is sufficiently low.
            if i > min_iterations and loss_mean < loss_limit:
                break

    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.

        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """

        # The tf.layers API uses this name for the weights in a conv-layer.
        variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable


    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights


    def get_layer_tensor(self, layer_name):
        """
        Return the tensor for the output of a layer.
        Note that this does not return the actual values,
        but instead returns a reference to the tensor
        inside the TensorFlow graph. Use get_tensor_value()
        to get the actual contents of the tensor.
        """

        # The name of the last operation of a layer,
        # assuming it uses Relu as the activation-function.
        tensor_name = layer_name + "/Relu:0"

        # Get the tensor with this name.
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.x: [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output


    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. Allows reloading of the counter
        with the tf checkpoint
        """
        return self.session.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_episodes_increase)

    def print_progress(self, msg):
        """
        Print progress on a single line and overwrite the line.
        Used during optimization.
        """

        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

  

