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


def model_predict(data, config_dict, local_model_path, pred_tensor_name):
    data_generator = SeriesDataGenerator(data, config_dict)
    saver = tf.train.import_meta_graph(model_meta_file(local_model_path))
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(local_model_path))
        # extract the placeholders and pred_op from model
        pred_op = sess.graph.get_tensor_by_name(pred_tensor_name)
        placeholders = _restore_placeholders(sess)
        feed_dict = _generate_feed(placeholders, data_generator, data_generator.total_row_counts, 1.)
        preds = sess.run(pred_op, feed_dict=feed_dict)
    # combine the prediction and target together
    combined_data = pd.DataFrame({'pageView': process_target_list(data.target),
                                  'pred_pageView': process_target_list(preds.tolist())})
    return combined_data


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


def _restore_placeholders(sess):
    placeholders = {}
    placeholders['input_x'] = sess.graph.get_tensor_by_name("placeholders/input_x:0")
    placeholders['input_meta_X'] = sess.graph.get_tensor_by_name("placeholders/input_meta_X:0")
    placeholders['input_y'] = sess.graph.get_tensor_by_name("placeholders/input_y:0")
    placeholders['dropout_input_keep_prob'] = sess.graph.get_tensor_by_name("placeholders/dropout_input_keep_prob:0")
    return placeholders

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
        combined_lstm_output = tf.concat([alll_units_output, placeholders['input_meta_X']], 1)
        mid_layer_nodes = 32
        combined_output = tflayers.stack(combined_lstm_output, tflayers.fully_connected, [mid_layer_nodes], scope='fully_connect_layer')
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


def train(placeholders, config, train_op, test_eval_op, data_generator, test_data_generator=None, USE_CPU=True):
    clear_folder(config.log_path)
    clear_folder(config.model_path)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    print 'models to be written into: ', config.model_path
    print 'logs to be written into: ', config.log_path
    start_time = time.time()
    writer = tf.summary.FileWriter(config.log_path)
    init = tf.global_variables_initializer()
    merged_summary_op = tf.summary.merge_all()
    generate_tensorboard_script(config.log_path)  # create the script to start a tensorboard session
    if USE_CPU:
        NUM_THREADS = multiprocessing.cpu_count()
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
    else:
        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.05

    with tf.Session(config=sess_config) as sess:
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


    #'''
    train_data = pd.read_csv(os.path.join(os.path.expanduser("~"), data_file), nrows=100)
    #train_data = pd.read_csv(os.path.join(os.path.expanduser("~"), data_file))
    print train_data.shape
    with open(os.path.join(os.path.expanduser("~"), yaml_file), 'r') as yaml_file:
        config_dict = yaml.load(yaml_file)
    #'''


    '''
    gcs_path = 'test/MachineLearning'
    index_gcs_path = 'test/MachineLearning/index_yaml'
    yaml_file_name = 'const_norm_binary_configuration.yaml'
    index_file_name = 'NYDN_hybrid_model_fold_2.yaml'

    config_dict, local_data_file = load_training_data_from_gcs(gcs_path, yaml_file_name)
    bucket = GCS_Bucket()
    index_dict = load_yaml_file_from_gcs(bucket, index_gcs_path, index_file_name)
    train_data, valid_data = create_train_test_by_index(local_data_file, config_dict, index_dict)
    '''
    data_generator = SeriesDataGenerator(train_data, config_dict)
    #test_data_generator = SeriesDataGenerator(valid_data, config_dict)

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

    placeholders = _init_placeholders(rnn_n_steps, rnn_input_length, meta_x_length)
    for key in placeholders:
        print 'the placeholder {} :'.format(key), placeholders[key]
    softmax_linear = _init_graph(placeholders, rnn_n_steps, rnn_hidden_state_length)
    print 'the pred op: ', softmax_linear
    loss = _compute_loss(softmax_linear, placeholders['input_y'])
    variable_summary(loss, 'training_loss')
    eval_op_list.append(loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train(placeholders, model_config, optimizer, loss, data_generator)


if __name__ == '__main__':
    main()


