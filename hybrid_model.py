import os, multiprocessing
import pandas as pd
from utils import clear_folder, model_meta_file, process_target_list
from create_tensorboard_start_script import generate_tensorboard_script
from series_data_generator import SeriesDataGenerator
from google_cloud_storage_util import GCS_Bucket


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers as tflayers
from tensorflow.python.client import device_lib


def model_predict(data, config_dict, local_model_path, pred_op_name):
    data_generator = SeriesDataGenerator(data, config_dict)
    data = data_generator.next_batch(data_generator.get_total_counts())
    saver = tf.train.import_meta_graph(model_meta_file(local_model_path))
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(local_model_path))
        # extract the placeholders and pred_op from model
        pred_op = sess.graph.get_tensor_by_name(pred_op_name)
        x = sess.graph.get_tensor_by_name("input_X:0")
        meta_x = sess.graph.get_tensor_by_name("input_meta_X:0")
        # run the pred_op in session
        preds = sess.run(pred_op, feed_dict={x: data.time_series_data,
                                             meta_x: data.meta_data})
    # combine the prediction and target together
    combined_data = pd.DataFrame({'pageView': process_target_list(data.target),
                                  'pred_pageView': process_target_list(preds.tolist())})
    return combined_data


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


class HybridModel(object):
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
    USE_CPU = True
    if len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0:
        USE_CPU = False

    def __init__(self, config_dict, model_name='hybrid_model', learning_rate=0.001, batch_size=20):
        # Parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = 1500
        #self.test_batch_size = 500
        self.display_step = 20
        self.gcs_bucket = GCS_Bucket()


        #self.n_hidden = 4  # hidden layer dimension
        #self.FC_layers = [1]

        self.n_hidden = 32  # hidden layer dimension
        self.FC_layers = [16, 1]
#        self.FC_layers = [8, 1]

        self.n_input = len(config_dict["time_interval_columns"])  # dimension of each time_step input
        self.n_meta_input = len(config_dict["static_columns"])  # dimension of meta input (categorical features)
        self.n_steps = len(config_dict["time_step_list"])  # time-steps in RNN
        self.model_name = model_name

        self.model_path = create_local_model_path(self.COMMON_PATH, self.model_name)
        self.log_path = create_local_log_path(self.COMMON_PATH, self.model_name)
        #self.log_path = os.path.join(self.model_path, 'log')
        generate_tensorboard_script(self.log_path)  # create the script to start a tensorboard session
        if self.USE_CPU:
            self.config = tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)
        else:
            self.config = tf.ConfigProto(log_device_placement=False)
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.08
            #self.config = tf.ConfigProto(log_device_placement=True,
            #                             gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.08))

        # model placeholders
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input], name='input_X')
        self.meta_x = tf.placeholder("float", [None, self.n_meta_input], name='input_meta_X')
        self.y = tf.placeholder("float", [None, 1], name='input_y')

    def get_model_path(self):
        return self.model_path

    @staticmethod
    def single_variable_summary(var, name):
        reduce_mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
        tf.summary.histogram('{}_histogram'.format(name), var)

    @staticmethod
    def variable_summaries(var, name):
        reduce_mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
        tf.summary.scalar('{}_max'.format(name), tf.reduce_max(var))
        tf.summary.scalar('{}_min'.format(name), tf.reduce_min(var))
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
            # combine the last LSTM unit output with `meta_X`
            combined_output = tf.concat([outputs[-1], meta_X], 1)

            # combine all the LSTM unit output with `meta_X`, similar to an attention model
            #alll_units_output = tf.concat([unit for unit in outputs], 1)
            #combined_output = tf.concat([alll_units_output, meta_X], 1)

            print 'combined output dimension: ', combined_output.shape
            output = tflayers.stack(combined_output, tflayers.fully_connected, layers, scope='fully_connect_layer')
            return output

    def build(self):
        # build the model
        self.pred = self.RNN(self.x, self.meta_x, self.model_name, self.FC_layers)
        print 'the predicting tensor: ', self.pred
        with tf.name_scope('loss'):
            #loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
            # reference to simplify the loss:
            # https://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
            #loss = tf.reduce_sum(tf.squared_difference(self.y, self.pred))  # the RMSE loss
            loss = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.pred)))  # the MAE loss

            self.single_variable_summary(loss, 'objective_func_loss')
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            print 'optimizer name: ',  optimizer.name
        return optimizer

    def create_eval_op(self, data_size, eval_name='eval'):
        with tf.name_scope(eval_name):
            #eval_op = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)  # the RMSE eval error
            eval_op = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.pred)) / data_size)  # the MAE loss
            self.single_variable_summary(eval_op, eval_name)
            #print 'eval_op name: ',  eval_op.name
        return eval_op

    def train(self, data_generator, test_data_generator=None):
        clear_folder(self.log_path)
        clear_folder(self.model_path)

        optimizer = self.build()  # the optimizer for model building
        if test_data_generator is not None:
            test_data_size = test_data_generator.get_total_counts()
        else:
            test_data_size = self.batch_size
        test_eval_op = self.create_eval_op(test_data_size, 'test_eval')  # eval operation using test data
        train_eval_op = self.create_eval_op(self.batch_size, 'train_eval')  # eval operation using train data
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        print 'models to be written into: ', self.model_path
        print 'logs to be written into: ', self.log_path
        writer = tf.summary.FileWriter(self.log_path)
        with tf.Session(config=self.config) as sess:
            # Launch the graph
            sess.run(init)
            step = 1
            train_MAE, test_MAE = "unavailable", "unavailable"
            writer.add_graph(sess.graph)
            '''
            with tf.name_scope('weight_matrix'):
                weight_matrix = sess.graph.get_tensor_by_name("fully_connect_layer/fully_connect_layer_2/weights:0")
                self.variable_summaries(weight_matrix, 'weight_matrix')
            '''
            merged_summary_op = tf.summary.merge_all()
            with tf.name_scope('training'):
                while step * self.batch_size < self.num_epochs * data_generator.get_total_counts():
                    data = data_generator.next_batch(self.batch_size)
                    sess.run(optimizer, feed_dict={self.x: data.time_series_data,
                                                   self.meta_x: data.meta_data,
                                                   self.y: data.target})

                    if step % self.display_step == 0:
                        # to validate using test data
                        if test_data_generator is not None:
                            # use all the test data every time
                            test_data = test_data_generator.next_batch(test_data_size)
                            summary, test_MAE = sess.run([merged_summary_op, test_eval_op],
                                                         feed_dict={self.x: test_data.time_series_data,
                                                                    self.meta_x: test_data.meta_data,
                                                                    self.y: test_data.target})

                            train_MAE = sess.run(train_eval_op,
                                                feed_dict={self.x: data.time_series_data,
                                                            self.meta_x: data.meta_data,
                                                            self.y: data.target})

                        else:
                            summary, train_MAE = sess.run([merged_summary_op, train_eval_op],
                                                          feed_dict={self.x: data.time_series_data,
                                                                     self.meta_x: data.meta_data,
                                                                     self.y: data.target})
                        writer.add_summary(summary, step)
                        saver.save(sess, os.path.join(self.model_path, 'models'), global_step=step)
                        print "Iter {}, train MAE: {}, test MAE: {}".format(step * self.batch_size,
                                                                            str(train_MAE),
                                                                            str(test_MAE))

                    step += 1
                saver.save(sess, os.path.join(self.model_path, 'final_model'), global_step=step)
                print "Optimization Finished!"



