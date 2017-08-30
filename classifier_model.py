import os, multiprocessing, time, yaml
import pandas as pd
from utils import clear_folder, model_meta_file, process_target_list
from create_tensorboard_start_script import generate_tensorboard_script
from series_data_generator import SeriesDataGenerator
from google_cloud_storage_util import GCS_Bucket
from data_preprocess import load_training_data_from_gcs, load_yaml_file_from_gcs, create_train_test_by_index


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers as tflayers


def model_predict(data, config_dict, local_model_path, pred_op_name):
    data_generator = SeriesDataGenerator(data, config_dict)
    data = data_generator.next_batch(data_generator.total_row_counts)
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



def _init_placeholders(rnn_n_steps, rnn_input_length, meta_x_length):
    placeholders = {}
    with tf.name_scope("placeholders"):
        placeholders['input_x'] = tf.placeholder("float", [None, rnn_n_steps, rnn_input_length], name='input_x')
        placeholders['input_meta_X'] = tf.placeholder("float", [None, meta_x_length], name='input_meta_X')
        placeholders['input_y'] = tf.placeholder("float", [None], name='input_y')
        placeholders['dropout_input_keep_prob'] = tf.placeholder(dtype=tf.float32, name='dropout_input_keep_prob')
    return placeholders


def _init_graph(placeholders, rnn_n_steps, rnn_hidden_state_length, num_class=5):
    with tf.name_scope('graph'):
        x = tf.unstack(placeholders['input_x'], rnn_n_steps, 1)
        # Define a LSTM cell
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_hidden_state_length)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=placeholders['dropout_input_keep_prob'])
        # Get LSTM cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, scope='LSTM_unit')

        # combine all the LSTM unit output with `meta_X`, similar to an attention model
        alll_units_output = tf.concat([unit for unit in outputs], 1)
        combined_output = tf.concat([alll_units_output, placeholders['input_meta_X']], 1)
        print 'combined output dimension: ', combined_output.shape
        output_length = combined_output.shape.as_list()[1]
        weights = tf.get_variable('weights',
                                  [output_length, num_class],
                                  initializer=tf.truncated_normal_initializer(stddev=1/(1.*output_length)),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(combined_output, weights), biases, name='softmax_linear')
        return softmax_linear


def _generate_feed(placeholders, data_generator, batch_size, dropout_input_keep_prob=1.):
        data = data_generator.next_batch(batch_size)
        return {placeholders['input_x']: data.time_series_data,
                placeholders['input_meta_X']: data.meta_data,
                placeholders['input_y']: data.target,
                placeholders['dropout_input_keep_prob']: dropout_input_keep_prob}


def _compute_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def train(placeholders, config, train_op, test_eval_op, data_generator, test_data_generator=None):
    clear_folder(config.log_path)
    clear_folder(config.model_path)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    print 'models to be written into: ', config.model_path
    print 'logs to be written into: ', config.log_path
    writer = tf.summary.FileWriter(config.log_path)
    init = tf.global_variables_initializer()
    start_time = time.time()
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        train_MAE, test_MAE = "unavailable", "unavailable"
        writer.add_graph(sess.graph)
        with tf.name_scope('training'):
            while step * config.batch_size < config.num_epochs * data_generator.total_row_counts:
                train_feed = _generate_feed(placeholders, data_generator, config.batch_size, config.dropout_input_keep_prob)
                _ = sess.run([train_op], feed_dict=train_feed)
                step += 1
                if step % config.display_step == 0:
                    if test_data_generator is not None:
                        summary, test_MAE = sess.run([merged_summary_op, test_eval_op],
                                                     feed_dict=_generate_feed(placeholders,
                                                                              test_data_generator,
                                                                              test_data_generator.total_row_counts,
                                                                              1.))
                    else:
                        summary, test_MAE = sess.run([merged_summary_op, test_eval_op],
                                                     feed_dict=_generate_feed(placeholders,
                                                                              data_generator,
                                                                              data_generator.total_row_counts,
                                                                              1.))
                    train_MAE = sess.run(test_eval_op, feed_dict=train_feed)
                    cur_time = time.time()
                    writer.add_summary(summary, step)
                    saver.save(sess, os.path.join(config.model_path, 'models'), global_step=step)
                    print "step {}, train MAE: {}, test MAE: {}, using {:.2f} seconds".format(step,
                                                                                              str(train_MAE),
                                                                                              str(test_MAE),
                                                                                              (cur_time - start_time))
                    start_time = cur_time
    print "Optimization Finished!"


def variable_summary(var, name):
    reduce_mean = tf.reduce_mean(var)
    tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
    tf.summary.histogram('{}_histogram'.format(name), var)

def main():
    data_file = 'tmp_data.csv'
    yaml_file = 'tmp_config.yaml'
    #train_data = pd.read_csv(os.path.join(os.path.expanduser("~"), data_file), nrows=100)
    train_data = pd.read_csv(os.path.join(os.path.expanduser("~"), data_file))
    print train_data.shape
    with open(os.path.join(os.path.expanduser("~"), yaml_file), 'r') as yaml_file:
        config_dict = yaml.load(yaml_file)

    '''
    gcs_path = 'test/MachineLearning'
    index_gcs_path = 'test/MachineLearning/index_yaml'
    yaml_file_name = 'const_norm_binary_configuration.yaml'
    index_file_name = 'NYDN_hybrid_model_fold_2.yaml'

    config_dict, local_data_file = load_training_data_from_gcs(gcs_path, yaml_file_name)
    bucket = GCS_Bucket()
    index_dict = load_yaml_file_from_gcs(bucket, index_gcs_path, index_file_name)
    train, valid_data = create_train_test_by_index(local_data_file, config_dict, index_dict)
    train.to_csv(os.path.join(os.path.expanduser("~"), data_file))
    with open(os.path.join(os.path.expanduser("~"), yaml_file), 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file)
    '''
    rnn_input_length = len(config_dict["time_interval_columns"])
    rnn_n_steps = len(config_dict["time_step_list"])
    meta_x_length = len(config_dict["static_columns"])
    rnn_hidden_state_length = 8
    learning_rate = 0.0001
    eval_op_list = []
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')
    class ClassifierModelConfig():
        pass
    model_config = ClassifierModelConfig()
    model_config.model_name = 'test_binary_model'
    model_config.batch_size = 256
    model_config.num_epochs = 5000
    model_config.display_step = 1000
    model_config.dropout_input_keep_prob = 0.8

    model_config.model_path = create_local_model_path(COMMON_PATH, model_config.model_name)
    model_config.log_path = create_local_log_path(COMMON_PATH, model_config.model_name)

    data_generator = SeriesDataGenerator(train_data, config_dict)
    placeholders = _init_placeholders(rnn_n_steps, rnn_input_length, meta_x_length)
    softmax_linear = _init_graph(placeholders, rnn_n_steps, rnn_hidden_state_length)
    loss = _compute_loss(softmax_linear, placeholders['input_y'])
    variable_summary(loss, 'training_loss')
    eval_op_list.append(loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train(placeholders, model_config, optimizer, loss, data_generator)


if __name__ == '__main__':
    main()


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
    NUM_THREADS = multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    def __init__(self, config_dict, model_name='hybrid_model', learning_rate=0.001, batch_size=20, USE_CPU=True):
        # Parameters
        self.USE_CPU = False
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = 10000
        #self.test_batch_size = 500
        self.display_step = 50
        self.gcs_bucket = GCS_Bucket()


        #self.n_hidden = 4  # hidden layer dimension
        #self.FC_layers = [1]

        self.n_hidden = 8  # hidden layer dimension
        self.FC_layers = [1]
        #self.FC_layers = [16, 1]

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
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        else:
            self.config = tf.ConfigProto(log_device_placement=False)
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.05

    def _init_placeholders(self):
        # initialize model placeholders
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input], name='input_X')
        self.meta_x = tf.placeholder("float", [None, self.n_meta_input], name='input_meta_X')
        self.y = tf.placeholder("float", [None, 1], name='input_y')
        self.dropout_input_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_input_keep_prob')
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

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

    def RNN(self):
        """ build the RNN model graph

        Given placeholders `X` and `meta_X`, as well as a list `layer` to
        represent structure of fully-connected part, to build a RNN model
        graph.

        expected Args:
            self.x (batch_size, n_steps, n_input): the model RNN part input
            mself.meta_x (batch_size, n_meta_input): the model fully-connected part input
            self.model_name (string) : model name
            self.FC_layers (List(int)): a list of integers to represent the Fully-connected part structure

         Returns:
            A `Tensor` with the dimension of layers[-1]
        """
        with tf.name_scope(self.model_name):
            # Unstack `X` by the axis of ``n_step`` to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(self.x, self.n_steps, 1)
            # Define a LSTM cell
            #lstm_cell = rnn.BasicLSTMCell(self.n_hidden, reuse=False)
            lstm_cell = tf.contrib.rnn.LSTMCell(self.n_hidden)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout_input_keep_prob)
            # Get LSTM cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, scope='LSTM_unit')

            # combine the last LSTM unit output with `meta_X`
            #combined_output = tf.concat([outputs[-1], self.meta_x], 1)

            # combine all the LSTM unit output with `meta_X`, similar to an attention model
            alll_units_output = tf.concat([unit for unit in outputs], 1)
            combined_output = tf.concat([alll_units_output, self.meta_x], 1)

            print 'combined output dimension: ', combined_output.shape
            output = tflayers.stack(combined_output, tflayers.fully_connected, self.FC_layers, scope='fully_connect_layer')
            return output

    def _init_optimizer(self, learning_rate):
        with tf.name_scope('loss'):
            # reference to simplify the loss:
            # https://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
            #loss = tf.reduce_sum(tf.squared_difference(self.y, self.pred))  # the RMSE loss
            tot_PV_sum = tf.reduce_sum(self.y) / 10.
            loss = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.pred)) / tot_PV_sum)  # the MAE loss
            self.single_variable_summary(loss, 'objective_func_loss')
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            print 'optimizer name: ',  optimizer.name
        return optimizer

    def create_eval_op(self, data_size, eval_name='eval'):
        with tf.name_scope(eval_name):
            #eval_op = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)  # the RMSE eval error
            eval_op = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.pred)) / data_size)  # the MAE loss
            self.single_variable_summary(eval_op, eval_name)
            #print 'eval_op name: ',  eval_op.name
        return eval_op

    def _generate_feed(self, data_generator, batch_size, dropout_input_keep_prob=1.):
        data = data_generator.next_batch(batch_size)
        return {self.x: data.time_series_data,
                self.meta_x: data.meta_data,
                self.y: data.target,
                self.dropout_input_keep_prob : dropout_input_keep_prob}

    def train(self, data_generator, test_data_generator=None, dropout_input_keep_prob=0.8):
        clear_folder(self.log_path)
        clear_folder(self.model_path)
        self._init_placeholders()
        # build the model
        # self.pred = self.RNN(self.x, self.meta_x, self.model_name, self.FC_layers)
        self.pred = self.RNN()
        print 'the predicting tensor: ', self.pred
        optimizer = self._init_optimizer(self.learning_rate)  # the optimizer for model building

        if test_data_generator is not None:
            test_data_size = test_data_generator.total_row_counts
        else:
            test_data_size = self.batch_size
        test_eval_op = self.create_eval_op(test_data_size, 'test_eval')  # eval operation using test data
        train_eval_op = self.create_eval_op(self.batch_size, 'train_eval')  # eval operation using train data
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        start_time = time.time()
        print 'models to be written into: ', self.model_path
        print 'logs to be written into: ', self.log_path
        writer = tf.summary.FileWriter(self.log_path)
        with tf.Session(config=self.config) as sess:
            print self.config.gpu_options
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
                while step * self.batch_size < self.num_epochs * data_generator.total_row_counts:
                    train_feed = self._generate_feed(data_generator, self.batch_size, dropout_input_keep_prob)
                    _, step = sess.run([optimizer, self.increment_global_step_op], feed_dict=train_feed)
                    if step % self.display_step == 0:
                        # to validate using test data
                        if test_data_generator is not None:
                            # use all the test data every time
                            summary, test_MAE = sess.run([merged_summary_op, test_eval_op],
                                                         feed_dict=self._generate_feed(test_data_generator,
                                                                                       test_data_generator.total_row_counts,
                                                                                       1.))
                            train_MAE = sess.run(train_eval_op, feed_dict=train_feed)
                        else:
                            summary, train_MAE = sess.run([merged_summary_op, train_eval_op], feed_dict=train_feed)
                        writer.add_summary(summary, step)
                        saver.save(sess, os.path.join(self.model_path, 'models'), global_step=step)
                        cur_time = time.time()
                        print "step {}, train MAE: {}, test MAE: {}, using {:.2f} seconds".format(step,
                                                                                                  str(train_MAE),
                                                                                                  str(test_MAE),
                                                                                                  (cur_time - start_time))
                        start_time = cur_time

                    step += 1
                saver.save(sess, os.path.join(self.model_path, 'final_model'), global_step=step)
                print "Optimization Finished!"



