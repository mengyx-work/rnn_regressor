import tensorflow as tf

def train_model(model, data_generator):
    model.build()
    model.eval_op()
    # Launch the graph
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        with tf.name_scope('training'):    
            while step * model.batch_size < model.training_iters:
                data = data_generator.next_batch(model.batch_size)
                sess.run(model.optimizer, feed_dict={model.x: data.time_series_data, model.meta_x: data.meta_data, model.y: data.target})
                if step % model.display_step == 0:
                    rmse = sess.run(model.eval_op, feed_dict={model.x: data.time_series_data, model.meta_x: data.meta_data, model.y: data.target})
                    print "Iter " + str(step * model.batch_size) + ", Minibatch rmse= " + str(rmse)
                step += 1
            print "Optimization Finished!"

