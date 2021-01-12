import pandas as pd
import numpy as np
import tensorflow as tf
import random
random.seed(202)
import time
import os

import data_input


START = time.time()
CUR = os.getcwd()
SUMMARY_DIR = os.path.join(CUR, 'summary')

norm, epsilon = False, 0.001
NUM_EPOCH = 5
query_bs = 100
L1_N = 400
L2_N = 120

#读取数据
file_train = os.path.join(CUR, 'oppo_search_round1/oppo_round1_train_20180929.txt')
file_vali = os.path.join(CUR, 'oppo_search_round1/oppo_round1_vali_20180929.txt')

data_train,vocab_map = data_input.get_data_bow(file_vali)
data_vali = data_input.get_data_bow(file_vali)
train_epoch_steps = int(len(data_train)/query_bs)-1
vali_epoch_steps = int(len(data_vali)/query_bs)-1
nwords = len(vocab_map)
def batch_normalization(x, phase_train, out_size):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0,shape=[out_size]),
                           name='beta',
                           trainable=True)
        gamma = tf.Variable(tf.constant(1.0,shape=[out_size]),
                            name='gamma',
                            trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')#计算均值和方差 axes:求解的维度
        ema = tf.train.ExponentialMovingAverage(decay=0.5) #指数加权平均的求法，具体的公式是 total=a*total+(1-a)*next,

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda : (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def add_layer(inputs, in_size, out_size, activation_function=None):
    wlimit = np.sqrt(6.0/(in_size+out_size))
    Weights = tf.Variable(tf.random_uniform([in_size, out_size],-wlimit, wlimit))
    biases = tf.Variable(tf.random_uniform([out_size],-wlimit, wlimit))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def get_cosine_score(query_arr, doc_arr):
    pooled_len1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr),1))
    pooled_len2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr),1))
    pooled_mul_l2 = tf.reduce_sum(tf.multiply(query_arr, doc_arr),1)
    cos_scores = tf.div(pooled_mul_l2, pooled_len1*pooled_len2+1e-8, name='cos_scores')
    return cos_scores


#构建模型
with tf.name_scope("input"):
    query_batch = tf.placeholder(tf.float32, shape=[None,None], name='query_batch')
    doc_batch = tf.placeholder(tf.float32, shape=[None,None], name='doc_batch')
    doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')

    on_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')
with tf.name_scope("FC1"):
    query_l1 = add_layer(query_batch, nwords, L1_N, activation_function=None)
    doc_l1 = add_layer(doc_batch, nwords, L1_N, activation_function=None)
with tf.name_scope("BN1"):
    query_l1 = batch_normalization(query_l1, on_train, L1_N)
    doc_l1 = batch_normalization(doc_l1, on_train, L1_N)
    query_l1 = tf.nn.relu(query_l1)
    doc_l1 = tf.nn.relu(doc_l1)
with tf.name_scope("Drop_out"):
    query_l1 = tf.nn.dropout(query_l1, keep_prob)
    doc_l1 = tf.nn.dropout(doc_l1, keep_prob)

with tf.name_scope("FC2"):
    query_l2 = add_layer(query_l1, L1_N, L2_N, activation_function=None)
    doc_l2 = add_layer(doc_l1, L1_N, L2_N, activation_function=None)

with tf.name_scope('BN2'):
    query_l2 = batch_normalization(query_l2, on_train, L2_N)
    doc_l2 = batch_normalization(doc_l2, on_train, L2_N)
    query_l2 = tf.nn.relu(query_l2)
    doc_l2 = tf.nn.relu(doc_l2)

    query_pred = tf.nn.relu(query_l2)
    doc_pred = tf.nn.relu(doc_l2)

#定义损失函数
with tf.name_scope("Cosine_Similarity"):
    cos_sim = get_cosine_score(query_pred, doc_pred)
    cos_sim_prod = tf.clip_by_value(cos_sim, 1e-8, 1.0) #可以将一个张量中的数值限制在一个范围之内。

with tf.name_scope("Loss"):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
    losses = tf.reduce_sum(cross_entropy)
    tf.summary.scalar('loss', losses)

with tf.name_scope("Training"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(losses)
    pass

merged = tf.summary.merge_all() #可以将所有summary全部保存到磁盘，以便tensorboard显示。
saver = tf.train.Saver() ## 创建一个Saver对象，选择性保存变量或者模型。


def pull_batch(data_map, batch_id):
    query, title, label, dsize = range(4)
    cur_data = data_map[batch_id * query_bs : (batch_id+1)*query_bs]
    query_in = [x[0] for x in cur_data]
    doc_in = [x[1] for x in cur_data]
    label = [x[2] for x in cur_data]
    return query_in, doc_in, label

def feed_dict(on_training, data_set, batch_id, drop_prob):
    query_in, doc_in, label = pull_batch(data_set, batch_id)
    query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
    return {query_batch: query_in, doc_batch:doc_in, doc_label_batch:label,
            on_train:on_training, keep_prob:drop_prob}

with tf.name_scope("Test"):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)
with tf.name_scope("Train"):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph) #指定一个文件用来保存图。

    start = time.time()
    for epoch in range(NUM_EPOCH):
        print(f'{epoch} training......')
        random.shuffle(data_train)
        for batch_id in range(train_epoch_steps):
            sess.run(train_step, feed_dict=feed_dict(True, data_train, batch_id, 0.5))
            pass
        end = time.time()

        #train loss
        epoch_loss = 0
        for i in range(train_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_train, i, 1))
            epoch_loss += loss_v
        epoch_loss /= train_epoch_steps
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss:epoch_loss})
        train_writer.add_summary(train_loss, epoch+1)
        print(f"{epoch} Train loss : {epoch_loss}, time:{end-start}")

        # # test loss
        # start = time.time()
        # epoch_loss = 0
        # for i in range(vali_epoch_steps):
        #     loss_v = sess.run(losses, feed_dict=feed_dict(False, data_vali, i, 1))
        #     epoch_loss += loss_v
        # epoch_loss /= (vali_epoch_steps)
        # test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        # train_writer.add_summary(test_loss, epoch + 1)
        # # test_writer.add_summary(test_loss, step + 1)
        # print("Epoch #%d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
        #       (epoch, epoch_loss, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)





