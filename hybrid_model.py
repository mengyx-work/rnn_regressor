import os, time
from utils import clear_folder
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers as tflayers


class hybrid_model(object):
    def __init__(self):
	# Parameters
        self.learning_rate = 0.0001
        self.num_epochs = 100
        self.batch_size = 1
        self.test_batch_size = 50
        self.display_step = 200

        self.n_input = 12  ## dimension of each time_step input
        self.n_meta_input = 7 ## dimension of meta input (categorical features)
        self.n_steps = 6 ## timesteps
        self.n_hidden = 32 # hidden layer num of features
        self.FC_layers = [16, 1]

        self.model_path = '/Users/matt.meng/Google_Drive/deep_learning/tensorflow/storage/'
        self.log_path = os.path.join(self.model_path, 'tensorflow_log')
        #self.model_path = '/Users/matt.meng/Google_Drive/deep_learning/tensorflow/tmp'
        #self.log_path = '/Users/matt.meng/Google_Drive/deep_learning/tensorflow/tmp_log/test'
        
        self.model_name = 'hybrid_model'
        NUM_THREADS = 5
        self.config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input], name='input_X')
        self.meta_x = tf.placeholder("float", [None, self.n_meta_input], name='input_meta_X')
        self.y = tf.placeholder("float", [None, 1], name='input_y')
        self.init = tf.global_variables_initializer()

    def variable_summaries(self, var, name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('{}_mean'.format(name), mean)
        #tf.summary.scalar('{}_max'.format(name), tf.reduce_max(var))
        #tf.summary.scalar('{}_min'.format(name), tf.reduce_min(var))
        tf.summary.histogram('{}_histogram'.format(name), var)

    def RNN(self, X, meta_X, model_name='TF_model', layers=[16, 1]):
        with tf.name_scope(model_name):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(X, self.n_steps, 1)
            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(self.n_hidden, reuse=False)
            # Get lstm cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, scope='LSTM_unit')
            # combine the LSTM unit output and meta data
            combined_output = tf.concat([outputs[-1], meta_X], 1)    
            print 'combined output dimension: ', combined_output.shape
            output = tflayers.stack(combined_output, tflayers.fully_connected, layers, scope='fully_connect_layer')
            return output

    '''
    def build(self):
        clear_folder(self.log_path)
        ## build the model
        tf.reset_default_graph()
        self.graph = tf.Graph().as_default() 
        with self.graph:
            self.pred = self.RNN(self.x, self.meta_x, self.model_name, self.FC_layers)
            with tf.name_scope('loss'):
                loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
                self.variable_summaries(loss, 'RMSE_loss')
            with tf.name_scope('optimizer'):
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
                #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
                self.optimizer = optimizer
            self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    '''


    def build(self):
        clear_folder(self.log_path)
        clear_folder(self.model_path)
        ## build the model
        self.pred = self.RNN(self.x, self.meta_x, self.model_name, self.FC_layers)
        #tf.identity(self.pred, name="pred_op")
        print 'the pred operator name: ', self.pred.name
        with tf.name_scope('loss'):
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
            self.variable_summaries(loss, 'RMSE_loss')
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            #print 'optimizer name: ',  optimizer.name
            self.optimizer = optimizer

    def creat_eval_op(self):
        with tf.name_scope('root_mean_square_error'):
            eval_op = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred))) / self.batch_size)
            self.variable_summaries(eval_op, 'RMSE_eval')
            #print 'eval_op name: ',  eval_op.name
            self.eval_op = eval_op

    def train(self, data_generator, test_data_generator=None):
        self.build()
        self.creat_eval_op()
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
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
                    sess.run(self.optimizer, feed_dict={self.x: data.time_series_data, self.meta_x: data.meta_data, self.y: data.target})
                    if step % self.display_step == 0:
                        summary, train_rmse = sess.run([merged, self.eval_op], feed_dict={self.x: data.time_series_data, self.meta_x: data.meta_data, self.y: data.target})
                        ## to validate using test data
                        if test_data_generator is not None:
                            test_data = test_data_generator.next_batch(self.test_batch_size)
                            summary, test_rmse = sess.run([merged, self.eval_op], 
                                                            feed_dict={ self.x: test_data.time_series_data, 
                                                                        self.meta_x: test_data.meta_data, 
                                                                        self.y: test_data.target})
                        writer.add_summary(summary, step)
                        self.saver.save(sess, self.model_path + 'tensorflow_model', global_step=step)

                        print "Iter {} Minibatch, train RMSE: {}, test RMSE: {}".format(step * self.batch_size, str(test_rmse), str(train_rmse))

                    step += 1
                print "Optimization Finished!"



