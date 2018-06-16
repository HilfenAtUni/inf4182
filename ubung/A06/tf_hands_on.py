import tensorflow as tf

sess = tf.Session()

# --------------------------------- Instructions-------------------------------#
# - Use only tensorflow functions for calculations.
# - Please clearly indicate the part number while printing the results.
# - When using random tensors as inputs, please also print the inputs as well.
# - Part (a) is already done for your reference.
# -----------------------------------------------------------------------------#

###############################################################################
# 1a (0 point): Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
x = tf.random_uniform([])
y = tf.random_uniform([])
result = tf.cond(x > y, lambda: tf.add(x, y), lambda: tf.subtract(x, y))
print('part (a): {}'.format(sess.run([x, y, result])))
###############################################################################

###############################################################################
# 1b (1 point): Create two 0-d tensors x and y from a normal distribution.
# Return x * y if x < y, x / y if x > y, x^2+y^2 otherwise.
# Hint: Look up tf.case().
###############################################################################
x_ = tf.random_normal([])
y_ = tf.random_normal([])
b1_ = lambda: tf.multiply(x_, y_)
b2_ = lambda: tf.divide(x_, y_)
b3_ = lambda: tf.add(tf.square(x_), tf.square(y_))
result_ = tf.case({tf.less(x_,y_):b1_, tf.greater(x_,y_):b2_}, default=b3_, exclusive=True)
data = sess.run([x_, y_, result_])

x, y, result = data[0], data[1], data[2]
print('1b.\nx:\n{}\ny:\n{}\nresult:\n{}\n'
      .format(x, y, result))
###############################################################################
# 1c (1 point): Create the tensor x of the value [[1, -2, -1], [0, 1, 2]]
# and y as a tensor of ones with the same shape as x.
# Return a boolean tensor that yields Trues if absolute value of x equals
# y element-wise.
# Hint: Look up tf.equal().
###############################################################################
x_ = tf.constant([[1, -2, -1], [0, 1, 2]])
y_ = tf.ones(tf.shape(x_), dtype=tf.int32)
result_ = tf.equal(x_, y_)
data = sess.run([x_, y_, result_])

x, y, result = data[0], data[1], data[2]
print('1c.\nx:\n{}\ny:\n{}\nresult:\n{}\n'
      .format(x, y, result))
###############################################################################
# 1d (1 point): Create a tensor x having 10 elements with random uniform numbers
# between -1 and 1
# Get the indices of elements in x which are postive.
# Hint: Use tf.where().
# Then extract elements whose values are positive.
# Hint: Use tf.gather().
###############################################################################
x_ = tf.random_uniform([10] , -1, 1, tf.float32)
idx_ = tf.where(tf.greater(x_, 0))
v_ = tf.gather(x_, idx_)
data = sess.run([x_, idx_, v_])

x, idx, v = data[0], data[1].T, data[2].T
print('1d.\nx:\n{}\nidx:\n{}\nv:\n{}\n'.format(x, idx, v))
###############################################################################
# 1e (2 point): Create two tensors x and y of shape 5 from any distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: Look up in TF documentation for methods to compute mean and sum
###############################################################################
x_ = tf.random_uniform([5], -1, 1)
y_ = tf.random_uniform([5], -1, 1)
mean_, small_ = tf.reduce_mean(tf.subtract(x_,y_)), tf.less(tf.reduce_mean(tf.subtract(x_,y_)), 0)
mse_ = tf.reduce_mean(tf.square(tf.subtract(x_,y_)))
sum_ = tf.reduce_sum(tf.abs(tf.subtract(x_,y_)))
result_ = tf.cond(small_, lambda: mse_, lambda: sum_)
data = sess.run([x_, y_, mean_, mse_, sum_, result_])

x, y, mean, mse, sum, result = data[0], data[1], data[2], data[3], data[4], data[5]
print('1e.\nx:\n{}\ny:\n{}\nmean:\n{}\nmse:\n{}\nsum:\n{}\nresult:\n{}\n'
      .format(x, y, mean, mse, sum, result))
###############################################################################
# 1f (2 pt): Create two random 2-d tensors x and y both of size 3 x 2.
# - Concatenate x and y in axis 0  if the sum of all elements of x is greater
#   than the sum of all elements of y
# - Otherwise, Concatenate x and y in axis 1
# Hint: Use tf.concat()
###############################################################################
x_ = tf.random_normal([3, 2])
y_ = tf.random_normal([3, 2])
xy0_ = tf.concat([x_, y_], axis=0)
xy1_ = tf.concat([x_, y_], axis=1)
sum_ = [tf.reduce_sum(x_), tf.reduce_sum(y_)]
result_ = tf.cond(tf.greater(sum_[0], sum_[1]), lambda:xy0_, lambda:xy1_)
data = sess.run([x_, y_, xy0_, xy1_, result_])

x, y, xy0, xy1, result = data[0], data[1], data[2], data[3], data[4]
print('1f.\nx:\n{}\ny:\n{}\nxy0:\n{}\nxy1:\n{}\nresult:\n{}'
      .format(x, y, xy0, xy1, result))
###############################################################################
# 1g (3 points): We want to find the pseudo inverse of a matrix A
# Create a 3x3 tensor A = [[1,2,3],[3,6,3],[7,8,9]]
# Find the transpose of A (Atrans)
# Calculate the matrix B = (Atrans x A)
# Take the inverse of B (Binv)
# Compute the pseudo inverse matrix A_pinv = Binv x Atrans
# Find the inverse of A (A_inv) and print both A_inv and A_pinv
###############################################################################
A_ = tf.constant([[1,2,3],[3,6,3],[7,8,9]], dtype=tf.float32)
Atrans_ = tf.transpose(A_)
B_ = tf.multiply(Atrans_, A_)
Binv_ = tf.matrix_inverse(B_)
A_pinv_ = tf.multiply(Binv_, Atrans_)
Ainv_ = tf.matrix_inverse(A_)
data = sess.run([A_, Ainv_, A_pinv_])

A, A_inv, A_pinv = data[0], data[1], data[2]
print('1g.\nA:\n{}\nA_inv:\n{}\nA_pinv:\n{}\n'
      .format(A, A_inv, A_pinv))
