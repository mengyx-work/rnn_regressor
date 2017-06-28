import os, time, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from series_data_generator import series_data_generator
from hybrid_model import hybrid_model
from utils import data_columns, train_data
from train_model import train_model

label_col = 'total_views'
index_col = 'articleId'
#dep_var_col = 'total_views'

data_path = '/Users/matt.meng/Google_Drive/Taboola/ML/'
processed_train_file = 'processed_train_data.csv'
processed_test_file = 'processed_test_data.csv'
train = pd.read_csv(os.path.join(data_path, processed_train_file))
test = pd.read_csv(os.path.join(data_path, processed_test_file))

independent_features = ['articleInfo_type', 'articleInfo_authorName', 'articleInfo_section', 'minLocalDateInWeek', 'minLocalTime', 'createTime', 'publishTime']
feature_name_strings = ['views_PageView', 'sessionReferrer_DIRECT_PageView', 'pageReferrer_OTHER_PageView', 'platform_PHON_PageView', 'platform_DESK_PageView', 'sessionReferrer_SEARCH_PageView', 'pageReferrer_SEARCH_PageView', 'platform_TBLT_PageView', 'pageReferrer_DIRECT_PageView', 'sessionReferrer_SOCIAL_PageView', 'pageReferrer_EMPTY_DOMAIN_PageView', 'pageReferrer_SOCIAL_PageView']
step_time_strings = ['0min_to_10min', '10min_to_20min', '20min_to_30min', '30min_to_40min', '40min_to_50min', '50min_to_60min']

columns = data_columns(step_time_strings = step_time_strings, 
                       feature_name_strings = feature_name_strings, 
                       meta_data_columns = independent_features, 
                       target_column = label_col)

data_generator = series_data_generator(train, columns)
'''
test_data = data_generator.next_batch(2000)
print np.asarray(test_data.meta_data).shape
print np.asarray(test_data.time_series_data).shape
'''

#tf.reset_default_graph()
model = hybrid_model()
#init = tf.initialize_all_variables()
optimizer = model.build()
eval_op = model.eval_op()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    ## training data
    sess.run(init)
    step = 1
    #writer.add_graph(sess.graph)
    # create log writer object
    with tf.name_scope('training'):    
        # Keep training until reach max iterations
        while step * model.batch_size < model.training_iters:
            data = data_generator.next_batch(model.batch_size)
            sess.run(optimizer, feed_dict={model.x: data.time_series_data, model.meta_x: data.meta_data, model.y: data.target})
            if step % model.display_step == 0:
                rmse = sess.run(eval_op, feed_dict={model.x: data.time_series_data, model.meta_x: data.meta_data, model.y: data.target})
                print "Iter " + str(step * model.batch_size) + ", Minibatch rmse= " + str(rmse)
            step += 1
        print "Optimization Finished!"
