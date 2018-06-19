import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Import data
# Note: this is deprecated is the newer tensorflow version, but it the easiest
# way to load the mnist dataset
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist', one_hot=True)


# Add arguments to this function to easily run different experiments
def experiment():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W0 = tf.get_variable('W0', shape=(784,100)) # the default initializer is good
    b0 = tf.get_variable('b0', shape=(100))
    z1 = tf.nn.relu(x @ W0 + b0) # if using python earlier than 3.5, replace x @ W0 with tf.matmul(x,W0)

    W1 = tf.get_variable('W1', shape=(100, 100))
    b1 = tf.get_variable('b1', shape=(100))
    z2 = tf.nn.relu(z1 @ W1 + b1)

    W2 = tf.get_variable('W2', shape=(100, 10))
    b2 = tf.get_variable('b2', shape=(10))
    y = z2 @ W2 + b2

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for s in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # Test trained model
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


experiment()
