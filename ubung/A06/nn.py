import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Import data
# Note: this is deprecated is the newer tensorflow version, but it the easiest
# way to load the mnist dataset
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.get_variable('W', shape=(784,10), initializer=tf.zeros_initializer)
b = tf.get_variable('b', shape=(10), initializer=tf.zeros_initializer)

y = tf.matmul(x, W) + b

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
for s in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if s % 100 == 0:
    acc, loss = sess.run([accuracy, cross_entropy],
                         feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("At Step {}: Accuracy = {}, Loss = {}".format(s, acc, loss))

# Test trained model
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
