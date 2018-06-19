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

def get_feature(data_feature, lastfri_date):
    te_feature = []
    tmp_tr = []
    for i in range(lastfri_date-9, lastfri_date+1, 1):
        tmp_tr.extend(data_feature[i])
    te_feature.append(tmp_tr)

    return te_feature

def RNN(X, weight, biases, feature_num, lstm_size):
    n_inputs = feature_num
    max_time = 10
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.matmul(final_state[1], weight) + biases

    return results

def ratio2value(y_prediction, te_label):
    result = []
    result_label = []
    y_prediction = y_prediction.tolist()
    score = 0
    up_cnt = 0
    down_cnt = 0

    for i in range(5):
        sign_pred = y_prediction[0][i*3:(i+1)*3].index(max(y_prediction[0][i*3:(i+1)*3]))
        if sign_pred == 1:
            result.append(-1)

        elif sign_pred == 2:
            result.append(0)

        else:
            result.append(1)
    for sign in result:
        if sign == 1:
            up_cnt += 1

        elif sign == -1:
            down_cnt += 1

    if max(up_cnt, down_cnt) == up_cnt:
            result = [1] * 5

    else:
        result = [-1] * 5
    print(result)
    return result

def date2index(fridate, fn):
    tmp = []
    workday = pd.read_csv('stock_workday/' + fn + '_workday.csv')
    tmp.extend(workday['date'])
    date_index = tmp.index(int(fridate))

    return date_index

def main():
    etf_id = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']
    lstm_size_list = [8, 8, 8, 15 ,8 ,8 ,8 ,15 ,8, 8 ,18 ,14 ,8, 15 ,5 ,15 ,15, 15]



    pred_date_list = ['20180110', '20180111', '20180112', '20180115', '20180116', '20180117', '20180118', '20180119', '20180122', '20180123', '20180124', '20180125', '20180126', '20180129', '20180130', '20180131', '20180201', '20180202', '20180205', '20180206', '20180207', '20180208', '20180209', '20180212', '20180221', '20180222', '20180223', '20180226', '20180227', '20180301', '20180302', '20180305', '20180306', '20180307', '20180308', '20180309', '20180312', '20180313', '20180314', '20180315', '20180316', '20180319', '20180320', '20180321', '20180322', '20180323', '20180326', '20180327', '20180328', '20180329', '20180330', '20180331', '20180402', '20180403', '20180409', '20180410', '20180411', '20180412', '20180413', '20180416', '20180417', '20180418', '20180419', '20180420', '20180423', '20180424', '20180425', '20180426', '20180427', '20180430', '20180502', '20180503', '20180504', '20180507', '20180508', '20180509', '20180510', '20180511', '20180514', '20180515', '20180516', '20180517', '20180518', '20180521', '20180522', '20180523', '20180524', '20180525', '20180528', '20180529', '20180530', '20180531', '20180601', '20180604', '20180605', '20180606', '20180607', '20180608']


    for fn in range(len(etf_id)):

        for pred_date in pred_date_list:
            print(etf_id[fn] + ' ' + pred_date)
            lstm_size = lstm_size_list[fn]
            tf.reset_default_graph()
            lastfri_date = date2index(pred_date, etf_id[fn])
            data_feature, feature_num = get_batch_feature(etf_id[fn])
            data_label_value, data_label_ratio = get_batch_label(etf_id[fn])
            te_feature = get_feature(data_feature, lastfri_date)

            lastfri_value = data_label_value[lastfri_date][0]
            te_label_value = data_label_value[lastfri_date-10:lastfri_date-1]



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
                predictions = ratio2value(y_prediction.eval(feed_dict = {x:te_feature}), te_label_value)
                ori_csv = pd.read_csv('ensemble_pred_result_2/result_' + pred_date + '.csv')
                title = list(ori_csv.columns.values)
                with open('ensemble_pred_result_2/result_' + pred_date + '.csv', 'w', newline='') as fout:
                    wr = csv.writer(fout)
                    wr.writerow(title)

                    for row in range(len(ori_csv)):
                        value = []
                        if int(ori_csv['ETFid'][row]) == int(etf_id[fn]):
                            value.append(etf_id[fn])
                            for column in range(5):
                                value.append(predictions[column])
                                value.append(ori_csv[title[(2*(column+1))]][row])

                        else:
                            for item in title:
                                value.append(ori_csv[item][row])

                        wr.writerow(value)


if __name__ == '__main__':

    main()
