import os, multiprocessing
from utils import clear_folder
from google_cloud_storage_util import GCS_Bucket
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers as tflayers


class hybrid_model(object):
    """the hybrid_model as a class

    model-independent Attributes:
        learning_rate (double):
        num_epochs (int) :
        test_patch_size (int) :
        display_step (int) :

    model-dependent Attributes:
        n_input (int) : the dimension of input vector, in this model

    """
    NUM_THREADS = 2 * multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    def __init__(self, config_dict, model_name='hybrid_model'):
        # Parameters
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.batch_size = 1
        self.test_batch_size = 50
        self.display_step = 20
        self.gcs_bucket = GCS_Bucket("newsroom-backend")

        self.n_hidden = 8  # hidden layer dimension
        self.FC_layers = [16, 1]

        self.n_input = len(config_dict["time_interval_columns"])  # dimension of each time_step input
        self.n_meta_input = len(config_dict["static_columns"])  # dimension of meta input (categorical features)
        self.n_steps = len(config_dict["time_step_list"])  # time-steps in RNN
        self.model_name = model_name

        self.model_path = os.path.join(self.COMMON_PATH, self.model_name)
        self.log_path = os.path.join(self.model_path, 'log')
        
        self.config = tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)
        # model placeholders
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input], name='input_X')
        self.meta_x = tf.placeholder("float", [None, self.n_meta_input], name='input_meta_X')
        self.y = tf.placeholder("float", [None, 1], name='input_y')
        self.init = tf.global_variables_initializer()

    @staticmethod
    def variable_summaries(var, name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_mean'.format(name), mean)
        #tf.summary.scalar('{}_max'.format(name), tf.reduce_max(var))
        #tf.summary.scalar('{}_min'.format(name), tf.reduce_min(var))
        tf.summary.histogram('{}_histogram'.format(name), var)

    def RNN(self, X, meta_X, model_name='TF_model', layers=[16, 1]):
        """ build the RNN model graph

        Given placeholders `X` and `meta_X`, as well as a list `layer` to
        represent structure of fully-connected part, to build a RNN model
        graph.

        Args:
            X (batch_size, n_steps, n_input): the model RNN part input
            meta_X (batch_size, n_meta_input): the model fully-connected part input
            model_name (string) : model name
            layers (List(int)): a list of integers to represent the Fully-connected part structure

         Returns:
            A `Tensor` with the dimension of layers[-1]
        """
        with tf.name_scope(model_name):
            # Unstack `X` by the axis of ``n_step`` to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(X, self.n_steps, 1)
            # Define a LSTM cell
            lstm_cell = rnn.BasicLSTMCell(self.n_hidden, reuse=False)
            # Get LSTM cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, scope='LSTM_unit')
            # combine the LSTM unit output with `meta_X`
            combined_output = tf.concat([outputs[-1], meta_X], 1)    
            print 'combined output dimension: ', combined_output.shape
            output = tflayers.stack(combined_output, tflayers.fully_connected, layers, scope='fully_connect_layer')
            return output

    def build(self):
        clear_folder(self.log_path)
        clear_folder(self.model_path)
        # build the model
        self.pred = self.RNN(self.x, self.meta_x, self.model_name, self.FC_layers)
        print 'the predict tensor: ', self.pred
        with tf.name_scope('loss'):
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
            self.variable_summaries(loss, 'RMSE_loss')
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            print 'optimizer name: ',  optimizer.name
            self.optimizer = optimizer

    def create_eval_op(self):
        with tf.name_scope('root_mean_square_error'):
            eval_op = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
            self.variable_summaries(eval_op, 'RMSE_eval')
            #print 'eval_op name: ',  eval_op.name
            self.eval_op = eval_op

    def train(self, data_generator, test_data_generator=None):
        self.build()
        self.create_eval_op()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        print 'models to be written into: ', self.model_path
        print 'logs to be written into: ', self.log_path
        merged = tf.summary.merge_all()  
        writer = tf.summary.FileWriter(self.log_path)
        with tf.Session(config=self.config) as sess:
            # Launch the graph
            sess.run(init)
            step = 1
            test_rmse = "unavailable"
            writer.add_graph(sess.graph)
            with tf.name_scope('training'):    
                while step * self.batch_size < self.num_epochs * data_generator.get_total_counts():
                    data = data_generator.next_batch(self.batch_size)
                    sess.run(self.optimizer, feed_dict={self.x: data.time_series_data,
                                                        self.meta_x: data.meta_data,
                                                        self.y: data.target})
                    if step % self.display_step == 0:
                        summary, train_rmse = sess.run([merged, self.eval_op], feed_dict={self.x: data.time_series_data,
                                                                                          self.meta_x: data.meta_data,
                                                                                          self.y: data.target})
                        # to validate using test data
                        if test_data_generator is not None:
                            test_data = test_data_generator.next_batch(self.test_batch_size)
                            summary, test_rmse = sess.run([merged, self.eval_op],
                                                          feed_dict={self.x: test_data.time_series_data,
                                                                     self.meta_x: test_data.meta_data,
                                                                     self.y: test_data.target})
                        writer.add_summary(summary, step)
                        saver.save(sess, os.path.join(self.model_path, 'tensorflow_model'), global_step=step)
                        print "Iter {} Minibatch, train RMSE: {}, test RMSE: {}".format(step * self.batch_size,
                                                                                        str(test_rmse),
                                                                                        str(train_rmse))

                    step += 1
                saver.save(sess, os.path.join(self.model_path, 'final_model'), global_step=step)
                print "Optimization Finished!"



