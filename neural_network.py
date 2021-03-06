import numpy as np
# import tensorflow as tf
import main
import sys
import os
import time
import csv
import argparse
import tensorflow as tf
# tf.disable_v2_behavior()

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

        self.training = training

        # Optional parameters
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model_name = model_name

            # Make tensorflow variable names
            name_q_values_new = model_name + "/q_values_new"
            name_count_states = model_name + "/count_states"
            name_count_episodes = model_name + "/count_episodes"
            name_layer_fc1 = model_name + "/layer_fc1"
            name_layer_fc2 = model_name + "/layer_fc2"
            name_layer_fc3 = model_name + "/layer_fc3"
            name_layer_fc4 = model_name + "/layer_fc4"
            name_layer_fc5 = model_name + "/layer_fc5"

            name_out_layer = model_name + "/layer_fc_out"
            name_learning_rate = model_name + "/learn_rate"

            self.replay_memory = replay_memory
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

            # Placeholder variable for input state 
            self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='x')
            self.loss_val = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name=model_name + "/loss_val")

            # Placeholder for the learning rate.
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name=name_learning_rate)
            self.max_test_acc = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name=model_name + "/test_acc")


            # self.learning_rate = 0.01

            # Placeholder variable for inputting the target Q-value that we want to estimate
            self.q_values_new = tf.placeholder(tf.float32,
                                               shape=[None, num_actions],
                                               name=name_q_values_new)

            # From Hvass:
            # This is a hack that allows us to save/load the counter for
            # the number of states processed in the game-environment.
            # We will keep it as a variable in the TensorFlow-graph
            # even though it will not actually be used by TensorFlow.
            self.count_states = tf.Variable(initial_value=0,
                                            trainable=False, dtype=tf.int64,
                                            name=name_count_states)


            # Similarly, this is the counter for the number of episodes.
            self.count_episodes = tf.Variable(initial_value=0,
                                              trainable=False, dtype=tf.int64,
                                              name=name_count_episodes)


            # TensorFlow operation for increasing count_states.
            self.count_states_increase = tf.assign(self.count_states,
                                                   self.count_states + 1)


            # TensorFlow operation for increasing count_episodes.
            self.count_episodes_increase = tf.assign(self.count_episodes,
                                                     self.count_episodes + 1)

            
            # Neural network architechture

            # Initialise weights to be close to zero.
            init = tf.truncated_normal_initializer(mean=0.00, stddev=2e-2)
            activation = tf.nn.tanh

            self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name=model_name + "/keep_prob")


            net = self.x

            hl1 = tf.layers.dense(inputs=self.x, name=name_layer_fc1, units=250, trainable=True,
                                  kernel_initializer=init, activation=activation)
            hl1 = tf.nn.dropout(hl1, keep_prob=self.keep_prob)

            # # First fully-connected (aka. dense) layer.
            hl2 = tf.layers.dense(inputs=hl1, name=name_layer_fc2, units=250, trainable=True,
                                  kernel_initializer=init, activation=activation)

            hl2 = tf.nn.dropout(hl2, keep_prob=self.keep_prob)

            # hl3 = tf.layers.dense(inputs=hl2, name=name_layer_fc3, units=350,
            #                       kernel_initializer=init, activation=activation)

            # hl3 = tf.nn.dropout(hl3, keep_prob=self.keep_prob)

            # # First fully-connected (aka. dense) layer.
            # hl3 = tf.layers.dense(inputs=hl2, name=name_layer_fc3, units=6,
            #                       kernel_initializer=init, activation=activation)
            # hl3 = tf.nn.dropout(hl3, keep_prob=keep_prob)


            # # First fully-connected (aka. dense) layer.
            # hl4 = tf.layers.dense(inputs=hl3, name=name_layer_fc4, units=12,
            #                       kernel_initializer=init, activation=activation)
            # hl4 = tf.nn.dropout(hl4, keep_prob=keep_prob)

            # # First fully-connected (aka. dense) layer.
            # hl5 = tf.layers.dense(inputs=hl4, name=name_layer_fc5, units=12,
            #                       kernel_initializer=init, activation=activation)
            # hl5 = tf.nn.dropout(hl5, keep_prob=keep_prob)

            # Final fully-connected layer.
            outl = tf.layers.dense(inputs=hl2, name=name_out_layer, units=num_actions,
                                  kernel_initializer=init, activation=None)

            net = outl

            # # First fully-connected (aka. dense) layer.
            # net = tf.layers.dense(inputs=net, name=name_layer_fc2, units=num_actions,
            #                       kernel_initializer=init, activation=activation)

            
            # The output of the Neural Network is the estimated Q-values
            # for each possible action in the game-environment.
            # RMSPropOptimizer is adaptive stochastic gradient descent, Root mean squared (RMS).
            self.q_values = net

            # L2-loss is the difference between
            # squared_error = tf.square(self.q_values - self.q_values_new[1])
            # sum_squared_error = tf.reduce_sum(squared_error, axis=1)

            sample_weights = np.array([0, 1])
            sample_weights = sample_weights.reshape(-1, 2)

            # self.loss = tf.losses.mean_squared_error(self.q_values, self.q_values_new)

            squared_error = tf.square(self.q_values - self.q_values_new)
            sum_squared_error = tf.reduce_sum(squared_error, axis=1)
            

            self.loss = tf.reduce_mean(sum_squared_error)

            # self.loss = tf.reduce_mean(sum_squared_error)
            # self.loss = tf.reduce_mean(tf.square(self.q_values - self.q_values_new))

            # Optimizer to minimise the loss function.
            # learning rate is a placeholder defined earlier, because this needs
            # to change dynamically as the optimization progresses
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # For saving checkpoints
            # if training:
            self.saver = tf.train.Saver()

            # Create a new TensorFlow session so we can run the Neural Network.
            self.session = tf.Session(graph=self.graph)




            # Load the most recent checkpoint if it exists,
            # otherwise initialize all the variables in the TensorFlow graph.
            self.load_checkpoint(checkpoint_dir)

            # self.session.run(self.max_test_acc, feed_dict={self.x_place: 10})

            # var = [v for v in tf.trainable_variables() if v.name == name_layer_fc2+"kernel:0"]

            # print(var)
            self.trainable_params = [v for v in tf.trainable_variables()]

            # if model_name == "bvb_6":
            # for v in tf.trainable_variables():
            #     print(self.session.run(v))


                # var = [v for v in tf.trainable_variables() if v.name == name_layer_fc2+"kernel:0"]

                # exit()
            # exit()



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
        feed_dict = {self.x: states, self.keep_prob: 1.0}


        # Calculate Q-values for the states
        Q_value_estimates = self.session.run(self.q_values, feed_dict=feed_dict)

        return Q_value_estimates


    def optimize(self, min_epochs=1.0, max_epochs=10, 
        batch_size=128, loss_limit=0.0015, learning_rate=1e-3):
        """
        Optimize nn by sampling states and Q-values from the replay memory.
        """
        # self.learning_rate = learning_rate

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
        if self.training:
            keep_prob = 1.0

        else:
            keep_prob = 1.0

        for i in range(max_iterations):
        # for i in range(int(1e6)):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            state_batch, q_values_batch = self.replay_memory.random_batch_v5()

            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.

            feed_dict = {self.x: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate,
                         self.keep_prob: keep_prob}


            # Perform one optimization step and get the loss-value.
            loss_val = self.session.run(self.loss, feed_dict=feed_dict)
            self.loss_val = tf.assign(self.loss_val, loss_val)
            self.session.run(self.loss_val)

            gradients = self.optimizer.compute_gradients(loss=self.loss)


            # gvs = self.optimizer.compute_gradients(self.loss_val, self.trainable_params[0])
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

            exit()

            print(gvs)


            # loss_val, _ = self.session.run([self.loss, self.optimizer],
            #                                feed_dict=feed_dict)


            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # loss_history.append(loss_val)

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            pct_epoch = i / iterations_per_epoch

            if i % 10 == 0:
                msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}\n"
                msg = msg.format(i, pct_epoch, loss_val, loss_mean)
                self.print_progress(msg)


            # if i > min_iterations and loss_mean < loss_limit:
            #     break

            prev_loss_val = loss_mean




        return loss_mean




            # # Stop the optimization if we have performed the required number
            # # of iterations and the loss-value is sufficiently low.
            # if i > min_iterations and loss_mean < loss_limit:
            #     break

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

    def update_max_test_acc(self, new_acc):
        curr_max_test_acc = self.session.run(self.max_test_acc)

        if new_acc >= curr_max_test_acc:
            print(curr_max_test_acc, new_acc)
            self.max_test_acc = tf.assign(self.max_test_acc, new_acc)
            self.session.run(self.max_test_acc)

            # self.max_test_acc = self.session.run(self.max_test_acc, feed_dict={self.max_test_acc: new_acc})
            return True

        elif curr_max_test_acc > 0.5:
            return False

        else:
            return True

  

