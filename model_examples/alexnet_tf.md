### Alexnet - Tensorflow
Tensorflow implementation of AlexNet for the task of estimating the probability distribution of correct orientations of an image. 
The input to the model consists of (277, 277) colored images with 3 channels (i.e. color bands).
The target is a continuous variable $y \in (0,2\pi)$ for image orientation. 

Please note that the attached implementations do not include code for generating training and testing sets.


```python
import numpy as np
import tensorflow as tf
import math
from deepcde.deepcde_tensorflow import cde_loss, cde_predict
from deepcde.utils import box_transform
from deepcde.bases.cosine import CosineBasis

# PRE-PROCESSING #############################################################################
# Create basis (in this case 31 cosine basis)
n_basis = 31
basis = CosineBasis(n_basis)

# ... Creation of training and testing set ...

# Evaluate the y_train over the basis
y_train = box_transform(y_train, 0, 2*math.pi)  # transform to a variable between 0 and 1
y_basis = basis.evaluate(y_train)               # evaluate basis
y_basis = y_basis.astype(np.float32)

# Features and basis are inserted in a dictionary of this form
features = {
    'x': tf.FixedLenFeature([20 * 5 * 1], tf.float32),
    'y': tf.FixedLenFeature([1], tf.float32),
    'y_basis': tf.FixedLenFeature([n_basis - 1], tf.float32)
}
# ... Generation of TensorFlow training and testing data ...

# ALEXNET DEFINITION #########################################################################
weight_sd = 0.01  # sd parameter for initialization of weights
marginal_beta = None  # Initialization parameter for bias in CDE layer

def model_function(features, y_basis, mode, 
                   weight_sd=weight_sd, marginal_beta=marginal_beta):
    # Input Layer
    input_layer = tf.reshape(features, shape=[-1, 277, 277, 3])

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        strides=[1, 4, 4, 1],
        inputs=input_layer,
        filters=64,
        kernel_size=[1, 11, 11, 1],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[5, 5],
        strides=[1, 1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layers 3, 4 and 5
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        strides=[1, 1, 1, 1],
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1, 1, 1],
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    pool5_flat = tf.reshape(pool5, [-1, 256 * 6 * 6])

    # Dense Layers
    dense_1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    dropout_1 = tf.layers.dropout(inputs=dense_1, rate=0.5,
                                  training=mode == tf.estimator.ModeKeys.TRAIN)

    dense_2 = tf.layers.dense(inputs=dropout_1, units=4096, activation=tf.nn.relu)
    dropout_2 = tf.layers.dropout(inputs=dense_2, rate=0.5,
                                  training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # CDE Layer
    beta = cde_layer(dropout_2, weight_sd, marginal_beta)

    # Loss Computation
    loss = cde_loss(beta, y_basis)
    metrics = {
        'cde_loss': tf.metrics.mean(cde_loss(beta, y_basis))
    }

    # Training and Evaluation Steps
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        # Get train operator, using Adam for instance
        train_op = tf.train.AdamOptimizer().minimize(
            loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    else:
        raise NotImplementedError('Unknown mode {}'.format(mode))

# TRAINING #################################################################################
cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
estimator = tf.estimator.Estimator(model_fn, model_dir, cfg)
# ... Creation of `train_set` and `test_set` objects ...
# ... Inclusion of all extra parameters like learning rate, momentum, etc. ...
tf.estimator.train_and_evaluate(estimator, train_set, test_set)


# PREDICTION ##############################################################################
# ... Selection of `x_test` to get conditional density estimate of ...
y_grid = np.linspace(0, 1, 1000)
beta = tf.placeholder(tf.float32, [1, n_basis-1])
with tf.Session() as sess:
    # ... Resume model from checkpoint ...
    cdes = cde_predict(sess, beta, 0, 1, y_grid, basis, n_basis,
                       input_dict={'features': x_test})
    cdes = cdes * 2 * math.pi  # Re-normalize
```