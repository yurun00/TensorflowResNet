import tensorflow as tf
import numpy as np
from ConvResCell import *

HParams = {'relu_leakiness': 0.1, 'num_classes': 1000, 'learning_rate': 0.01}

class ResNet(object):
    def __init__(self, hps, images, labels):
        self.hps = hps
        self.images = images
        self.labels = labels
        self.inference()

    def inference(self, input_flow=None):
        if input_flow is None:
            input_flow = self.images
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.variable_scope('input_conv') as scope:
            n = 7 * 7 * 64
            conv_in = tf.layers.conv2d(input_flow, name='conv_in', filters=64, kernel_size=7,strides=(2, 2),
                                       padding='same', activation=tf.nn.relu, reuse=None,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
                                       bias_initializer=tf.constant_initializer())
        # max pool layer
        pool_in = tf.nn.max_pool(conv_in, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool_in')

        # residual parts
        res_1 = ConvResCell(3, 'res_1', hps=self.hps)
        sub_res_1 = res_1(pool_in, stride=1, subscope='sub_1')
        sub_res_2 = res_1(sub_res_1, stride=1, subscope='sub_2')
        sub_res_3 = res_1(sub_res_2, stride=1, subscope='sub_3')

        input_shape = sub_res_3.get_shape().as_list()
        pool_out = tf.nn.avg_pool(sub_res_3, ksize=[1, input_shape[1], input_shape[2], 1], strides=[1,1,1,1],
                                padding='VALID', name='pool_out')

        with tf.variable_scope('softmax'):
            logits = tf.layers.dense(inputs=pool_out, units=self.hps['num_classes'])
            softmax = tf.nn.softmax(logits)

        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.summary.scalar('loss', self.loss)


    def train(self):
        with tf.variable_scope('train'):
            self.lr = self.hps['learning_rate']
            grads = tf.gradients(self.loss, tf.trainable_variables())

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            grads = optimizer.compute_gradients(self.loss)
            apply_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        return apply_op



    def _fully_connected(self, input_flow, output_dim):
        x = tf.reshape(input_flow, [input_flow.get_shape()[0], -1])
        w = tf.get_variable(
            'weight', [x.get_shape()[1], output_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [output_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(x, w, b)

def build_model():
    with tf.Graph().as_default():
        input_image = tf.Variable(tf.zeros([1, 128, 128, 3]), dtype=tf.float32, name='input_image')
        label = tf.Variable(tf.zeros([1000]), dtype=tf.float32, name='label')
        rn = ResNet(HParams, input_image, label)
        train_step = rn.train()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('log/build_model/ResNet/', sess.graph)

        merged = tf.summary.merge_all()
        summary, _ = sess.run([merged, train_step])
        summary_writer.add_summary(summary, 1)

def main(_):
    build_model()

if __name__ == '__main__':
    tf.app.run()
