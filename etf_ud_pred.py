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
    df_value = pd.read_csv('stock_label/' + fn + '_label_ud.csv', encoding='utf-8')
    ud_label = []
    for row in range(len(df_value)):
        if df_value[fn + '_收盤價(元)'][row] == 1:
            ud_label.append([1, 0, 0])
        elif df_value[fn + '_收盤價(元)'][row] == -1:
            ud_label.append([0, 1, 0])
        else:
            ud_label.append([0, 0, 1])

    return ud_label

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

def RNN(X, weight, biases, feature_num, lstm_size):
    n_inputs = feature_num
    max_time = 10
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.nn.relu(tf.matmul(final_state[1], weight) + biases)

    return results

def ratio2value(y_prediction, te_label):
    result = []
    result_label = []
    y_prediction = y_prediction.tolist()
    score = 0

    for i in range(5):
        sign_pred = y_prediction[0][i*3:(i+1)*3].index(max(y_prediction[0][i*3:(i+1)*3]))
        if sign_pred == 1:
            result.append(-1)

        elif sign_pred == 2:
            result.append(0)

        else:
            result.append(1)

    for i in range(5):
        sign_true = te_label[i].index(1)
        if sign_true == 1:
            result_label.append(-1)

        elif sign_true == 2:
            result_label.append(0)

        else:
            result_label.append(1)

    for i in range(len(result)):
        tmp_score = 0
        if result[i] == result_label[i]:
            tmp_score += 0.5
        tmp_score = tmp_score * ((i+1)*0.05+0.05)
        score += tmp_score

    print('true:')
    print(result_label)
    print('pred:')
    print(result)
    return score


def main():
    fname = '713'
    lstm_size = 6


    data_feature, feature_num = get_batch_feature(fname)
    data_label_ud = get_batch_label(fname)
    tr_feature, te_feature = get_feature(data_feature)
    tr_label = get_tr_label(data_label_ud)

    te_label_value = data_label_ud[len(data_label_ud)-5:len(data_label_ud)]



    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 10*feature_num], name = 'x_input')
        y = tf.placeholder(tf.float32, [None, 15], name = 'y_input')

    with tf.name_scope('Weight'):
        weights = tf.Variable(tf.truncated_normal([lstm_size, 15], stddev = 0.1))

    with tf.name_scope('bias'):
        biases = tf.Variable(tf.constant(0.1, shape = [1]))

    with tf.name_scope('Layer'):
        y_prediction = RNN(x, weights, biases, feature_num, lstm_size)

    with tf.name_scope('avg_cost'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_prediction, labels = y))
        tf.summary.scalar('avg_cost', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        #accuracy =  tf.matmul(tf.divide(tf.subtract(y, tf.abs(tf.subtract(y_prediction, y))), y))
        accuracy  = y_prediction
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()


    with tf.Session(config=tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)
        batch_size = 50
        batch_num = len(tr_feature) // batch_size

        for epoch in range(500):
            for batch_i in range(batch_num):
                try:
                    batch_xs = tr_feature[batch_i*batch_size: (batch_i+1)*batch_size]
                    batch_ys = tr_label[batch_i*batch_size: (batch_i+1)*batch_size]
                except:
                    batch_xs = tr_feature[batch_i*batch_size:]
                    batch_ys = tr_label[batch_i*batch_size:]

                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})



            #testing_acc, test_res = sess.run([accuracy, merged], feed_dict={x:test_data, y:test_label})
            if epoch % 50 == 0 or epoch == 499:
                predictions = ratio2value(y_prediction.eval(feed_dict = {x:te_feature}), te_label_value)
                print('Epoch: ' + str(epoch) + ' score: ')
                print(predictions)
                print('')

        saver.save(sess, 'etf_ud_net/'+ fname + '_net/' + fname + '.ckpt')



if __name__ == '__main__':
    main()
