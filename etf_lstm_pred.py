import tensorflow as tf
import pandas as pd
import numpy as np

def get_batch_feature():
    df = pd.read_csv('stock_feature.csv')
    title = list(df.columns.values)
    data = []
    for row in range(len(df)):
        tmp_list = []
        for item in title:
            tmp_list.append(df[item][row])
        data.append(tmp_list)
    return data

def get_batch_label():
    #title = ['50_收盤價(元)', '51_收盤價(元)', '52_收盤價(元)', '53_收盤價(元)', '54_收盤價(元)', '55_收盤價(元)', '56_收盤價(元)', '57_收盤價(元)', '58_收盤價(元)', '59_收盤價(元)', '6201_收盤價(元)', '6203_收盤價(元)', '6204_收盤價(元)', '6208_收盤價(元)', '690_收盤價(元)', '692_收盤價(元)', '701_收盤價(元)', '713_收盤價(元)']
    title = ['50_收盤價(元)']
    df = pd.read_csv('stock_feature.csv')
    data = []
    for row in range(len(df)):
        tmp_list = []
        for item in title:
            tmp_list.append(df[item][row])
        data.append(tmp_list)
    return data

def RNN(X, weight, biases):
    n_inputs = 36
    max_time = 7
    lstm_size = 2
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    #final_state[0] is cell state ~ value in memory cell
    #final_state[1] is hidden_state ~ value of h'(memory cell)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weight) + biases)
    return results

def main():
    data_feature = get_batch_feature()
    data_label = get_batch_label()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [7, 36], name = 'x_input')
        y = tf.placeholder(tf.float32, [1, 1], name = 'y_input')

    with tf.name_scope('Weight'):
        weights = tf.Variable(tf.truncated_normal([2, 1], stddev = 0.1))

    with tf.name_scope('bias'):
        biases = tf.Variable(tf.constant(0.1, shape = [1]))

    with tf.name_scope('Layer'):
        y_prediction = RNN(x, weights, biases)

    with tf.name_scope('avg_cost'):
        mse = tf.losses.mean_squared_error(y_prediction, y)
        tf.summary.scalar('avg_cost', mse)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.2).minimize(mse)

    with tf.name_scope('accuracy'):
        #accuracy =  tf.subtract(y, tf.divide(tf.abs(tf.subtract(y_prediction, y)), y))*0.5
        accuracy = y_prediction
    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)
        test_data = []
        test_label = []

        for batch_i in range(len(data_feature)-8, len(data_feature)-1, 1):
            test_data.append(data_feature[batch_i])
            if batch_i == len(data_feature)-2:
                test_label = data_label[batch_i+1]
        test_label = np.asarray([test_label])

        for epoch in range(50):
            for batch_i in range(0, len(data_feature)-8, 1):
                batch_xs = []
                batch_ys = []
                for weekd in range(batch_i, batch_i+7, 1):
                    batch_xs.append(data_feature[weekd])
                    if weekd == batch_i + 6:
                        batch_ys = data_label[weekd+1]
                batch_xs = np.asarray(batch_xs)
                batch_ys = np.asarray([batch_ys])
                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
                testing_acc, test_res = sess.run([accuracy, merged], feed_dict={x:batch_xs, y:batch_ys})
                print('testing acc: ' + str(testing_acc))

            #testing_acc, test_res = sess.run([accuracy, merged], feed_dict={x:test_data, y:test_label})
            test_writer.add_summary(test_res, epoch)
            print('testing acc: ' + str(testing_acc))


if __name__ == '__main__':
    tf.reset_default_graph()
    main()
