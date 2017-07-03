import tensorflow as tf
import numpy as np

class ConvResCell(object):
    def __init__(self, kernel_size, scope, hps=None):
        """

        :param kernel_size: [height, width]
        :param scope: scope name
        :param hps: hyper-parameters
        """
        self.hps = hps
        self.kernel_size = kernel_size
        self.num_Conv2d = 2
        self.scope = scope

    def __call__(self, input_flow, stride, subscope):
        # input_shape: [batch, height, width, channels]
        input_shape = input_flow.get_shape().as_list()
        # output_shape: [batch, height, width, channels]
        output_shape = [input_shape[0], input_shape[1] // stride,
                        input_shape[2] // stride, input_shape[3] * (stride ** 2)]
        # filter strides of first convolutional layer
        strides = [1, stride, stride, 1]

        with tf.variable_scope(self.scope + "/" + subscope or type(self).__name__) as scope:
            identity = input_flow
            if input_shape[3] != output_shape[3]:
                identity = tf.nn.avg_pool(input_flow, stride, stride, 'VALID')
                identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0],
                                             [output_shape[3]//2, output_shape[3]//2]], )
                identity = self._conv('conv_s', input_flow, 1, input_shape[3],
                                      output_shape[3], strides)

            conv_1 = self._conv('conv_1', input_flow, self.kernel_size, input_shape[3],
                                output_shape[3], strides)
            hidden_1 = self._relu(conv_1, self.hps['relu_leakiness'])
            # hidden_1 = tf.nn.relu(conv_1, name=scope.name)
            conv_2 = self._conv('conv_2', hidden_1, self.kernel_size, output_shape[3],
                                output_shape[3], [1, 1, 1, 1])

            hidden_2 = self._relu(conv_2 + identity, self.hps['relu_leakiness'])
            # hidden_2 = tf.nn.relu(conv_2 + identity, name=scope.name)

        return hidden_2

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _conv(self, scope, x, kernel_size, in_channels, out_channels, strides):
        with tf.variable_scope(scope):
            n = kernel_size * kernel_size * out_channels
            filter = tf.get_variable(
                'filter', [kernel_size, kernel_size, in_channels, out_channels],
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
                dtype = tf.float32)
            conv = tf.nn.conv2d(x, filter, strides, padding='SAME')
            biases = tf.get_variable('biases', [out_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            return pre_activation