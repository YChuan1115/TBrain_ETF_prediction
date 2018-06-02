import tensorflow as tf
import pandas as pd
import numpy as np
import csv
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

def RNN(X, weight, biases, feature_num, lstm_size):
    n_inputs = feature_num
    max_time = 10
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.matmul(final_state[1], weight) + biases

    return results

def ratio2value(y_prediction, day_l5, te_label):
    result = []
    score = 0
    csv_result = []

    for i in range(5):
        if i == 0:
            result.append(float('%.2f' % ((y_prediction[0][i] * day_l5) + day_l5)))
        else:
            result.append(float('%.2f' % ((y_prediction[0][i] * result[i-1]) + result[i-1])))

    for i in range(len(result)):
        tmp_score = 0
        tmp_score = ((te_label[i+1][0] - abs(result[i] - te_label[i+1][0])) / te_label[i+1][0]) * 0.5
        if np.sign([te_label[i+1][0] - te_label[i][0]]) == np.sign([y_prediction[0][i]]):
            tmp_score += 0.5
        tmp_score = tmp_score * ((i+1)*0.05+0.05)
        score += tmp_score

    for i in range(len(result)):
        csv_result.append(int(np.sign([y_prediction[0][i]])))
        csv_result.append(result[i])

    return csv_result

def date2index(fridate, fn):
    tmp = []
    workday = pd.read_csv('stock_workday/' + fn + '_workday.csv')
    tmp.extend(workday['date'])
    date_index = tmp.index(int(fridate))

    return date_index

def main():
    etf_id = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    lstm_size_list = [11, 4, 2, 25, 8, 26, 3, 7, 6, 4, 2, 8, 8, 8, 19, 10, 7, 6]
    pred_date_list = ['20180112',
                     '20180119',
                     '20180126',
                     '20180202',
                     '20180209',
                     '20180226',
                     '20180306',
                     '20180313',
                     '20180320',
                     '20180327',
                     '20180402',
                     '20180412',
                     '20180419',
                     '20180426',
                     '20180504',
                     '20180511',
                     '20180518',
                     '20180525',
                     '20180601']
    #pred_date = '20180518'

    for fn in range(len(etf_id)):

        for pred_date in pred_date_list:
            print(etf_id[fn] + ' ' + pred_date)
            lstm_size = lstm_size_list[fn]
            tf.reset_default_graph()
            lastfri_date = date2index(pred_date, etf_id[fn])
            data_feature, feature_num = get_batch_feature(etf_id[fn])
            data_label_value, data_label_ratio = get_batch_label(etf_id[fn])
            tr_feature, te_feature = get_feature(data_feature)

            lastfri_value = data_label_value[lastfri_date][0]
            te_label_value = data_label_value[lastfri_date-10:lastfri_date-1]



            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, [None, 10*feature_num], name = 'x_input')
                y = tf.placeholder(tf.float32, [None, 5], name = 'y_input')

            with tf.name_scope('Weight'):
                weights = tf.Variable(tf.truncated_normal([lstm_size, 5], stddev = 0.1))

            with tf.name_scope('bias'):
                biases = tf.Variable(tf.constant(0.1, shape = [1]))

            with tf.name_scope('Layer'):
                y_prediction = RNN(x, weights, biases, feature_num, lstm_size)

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
            saver = tf.train.Saver()


            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, 'etf_ud_net/'+ etf_id[fn] + '_net/' + etf_id[fn] + '.ckpt')
                predictions = ratio2value(y_prediction.eval(feed_dict = {x:te_feature}), lastfri_value, te_label_value)
                ori_csv = pd.read_csv('pred_result/result_' + pred_date + '.csv')
                title = list(ori_csv.columns.values)
                with open('pred_result/result_' + pred_date + '.csv', 'w', newline='') as fout:
                    wr = csv.writer(fout)
                    wr.writerow(title)

                    for row in range(len(ori_csv)):
                        value = []
                        if row == etf_id.index(etf_id[fn]):
                            value.append(etf_id[fn])
                            value.extend(predictions)

                        else:
                            for item in title:
                                value.append(ori_csv[item][row])

                        wr.writerow(value)


if __name__ == '__main__':

    main()
