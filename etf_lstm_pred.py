import tensorflow as tf
import pandas as pd
import numpy as np
import math

def get_batch_feature(fn):
    df = pd.read_csv('stock_feature/'+ fn + '_feature.csv')
    title = list(df.columns.values)
    feature_num = len(title)
    data = []
    for row in range(len(df)):
        tmp_list = []
        for item in title:
            tmp_list.append(df[item][row])
        data.append(tmp_list)
    return data, feature_num

def get_batch_label(fn):
    title = [fn + '_收盤價(元)']
    df_value = pd.read_csv('stock_label/' + fn + '_label_value.csv')
    df_ratio = pd.read_csv('stock_label/' + fn + '_label_ratio.csv')
    value_label = []
    ratio_label = []
    for row in range(len(df_value)):
        tmp_list_value = []
        tmp_list_ratio = []
        for item in title:
            tmp_list_value.append(df_value[item][row])
            tmp_list_ratio.append(df_ratio[item][row])
        value_label.append(tmp_list_value)
        ratio_label.append(tmp_list_ratio)

    return value_label, ratio_label

def get_feature(data_feature):
    tr_feature = []
    te_feature = []
    for i in range(0, len(data_feature)-14, 1):
        tmp_tr = []
        for weekd in range(i, i+10, 1):
            tmp_tr.extend(data_feature[weekd])
        tr_feature.append(tmp_tr)

    for i in range(len(data_feature)-10, len(data_feature), 1):
        te_feature.extend(data_feature[i])

    return tr_feature, [te_feature]

def get_tr_label(data_label_value):
    tr_label = []
    for i in range(10, len(data_label_value)-4, 1):
        tmp_label = []
        for weekd in range(i, i+5, 1):
            tmp_label.extend(data_label_value[weekd])
        tr_label.append(tmp_label)

    return tr_label

def RNN(X, weight, biases, feature_num):
    n_inputs = feature_num
    max_time = 10
    lstm_size = 30
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.matmul(final_state[1], weight) + biases

    return results

def ratio2value(y_prediction, day_l5):
    result = []
    for i in range(5):
        if i == 0:
            result.append(float('%.2f' % ((y_prediction[0][i] * day_l5) + day_l5)))
        else:
            result.append(float('%.2f' % ((y_prediction[0][i] * result[i-1]) + result[i-1])))

    return result

def main():
    fname = '50'
    data_feature, feature_num = get_batch_feature(fname)
    data_label_value, data_label_ratio = get_batch_label(fname)
    tr_feature, te_feature = get_feature(data_feature)
    tr_label = get_tr_label(data_label_ratio)
    lastfri_value = data_label_value[len(data_label_value)-1][0]



    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 10*feature_num], name = 'x_input')
        y = tf.placeholder(tf.float32, [None, 5], name = 'y_input')

    with tf.name_scope('Weight'):
        weights = tf.Variable(tf.truncated_normal([30, 5], stddev = 0.1))

    with tf.name_scope('bias'):
        biases = tf.Variable(tf.constant(0.1, shape = [1]))

    with tf.name_scope('Layer'):
        y_prediction = RNN(x, weights, biases, feature_num)

    with tf.name_scope('avg_cost'):
        #score = tf.divide(tf.subtract(tf.abs(y), tf.abs(tf.subtract(y_prediction, y))), tf.abs(y))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, y_prediction))))
        tf.summary.scalar('avg_cost', rmse)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(rmse)

    with tf.name_scope('accuracy'):
        #accuracy =  tf.matmul(tf.divide(tf.subtract(y, tf.abs(tf.subtract(y_prediction, y))), y))
        accuracy  = y_prediction
    merged = tf.summary.merge_all()


    with tf.Session(config=tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)
        batch_size = 50
        batch_num = len(tr_feature) // batch_size

        for epoch in range(1000):
            for batch_i in range(batch_num):
                try:
                    batch_xs = tr_feature[batch_i*batch_size: (batch_i+1)*batch_size]
                    batch_ys = tr_label[batch_i*batch_size: (batch_i+1)*batch_size]
                except:
                    batch_xs = tr_feature[batch_i*batch_size:]
                    batch_ys = tr_label[batch_i*batch_size:]

                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})



            #testing_acc, test_res = sess.run([accuracy, merged], feed_dict={x:test_data, y:test_label})
            if epoch % 50 == 0 or epoch == 999:
                predictions = ratio2value(y_prediction.eval(feed_dict = {x:te_feature}), lastfri_value)
                print('Epoch: ' + str(epoch) + ' prediction: ')
                print(predictions)
                print('')


if __name__ == '__main__':
    main()
