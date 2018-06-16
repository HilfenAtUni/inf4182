import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Import data
# Note: this is deprecated is the newer tensorflow version, but it the easiest
# way to load the mnist dataset
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True)
img, lab = mnist.train.next_batch(100)
print('mnist:images:{}, labels:{}, test:{}'.format(img.shape, lab.shape, mnist.test.images.shape))
# Create the model
x = tf.placeholder(tf.float32, [None, 784])

# W = tf.get_variable('W', shape=(784,10), initializer=tf.zeros_initializer)
# b = tf.get_variable('b', shape=(10), initializer=tf.zeros_initializer)
# y = tf.matmul(x, W) + b

# c. f2(f1(x*w1+b1)*w2+b2)*w3+b3
idx = [20, 20] # [10, 10], [50,50], [100,100]
w1 = tf.Variable(tf.truncated_normal(shape=[784,idx[0]]))
b1 = tf.get_variable('b1', shape=(idx[0]), initializer=tf.zeros_initializer)
w2 = tf.Variable(tf.truncated_normal(shape=[idx[0],idx[1]]))
b2 = tf.get_variable('b2', shape=(idx[1]), initializer=tf.zeros_initializer)
w3 = tf.Variable(tf.truncated_normal(shape=[idx[1],10]))
b3 = tf.get_variable('b3', shape=(10), initializer=tf.zeros_initializer)
# bulid layers
f1 = tf.nn.relu(tf.matmul(x, w1) + b1)
f2 = tf.nn.relu(tf.matmul(f1, w2) + b2)
y = tf.nn.relu(tf.matmul(f2, w3) + b3)
# y = (tf.matmul(f2, w3) + b3)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# Accuracy for testing purposes
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tf.nn.softmax_cross_entropy_with_logits is a numerically stabilized softmax-
# cross-entropy, the logits are the preactivations calculated by the net
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
for s in range(20000): # 10000, 20000, 25000, 30000, 35000, 50000
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if s % 100 == 0:
      acc, loss = sess.run([accuracy, cross_entropy],
                           feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print("At Step {}: Accuracy = {}, Loss = {}".format(s, acc, loss))

# Test trained model
print('accuracy: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})))
