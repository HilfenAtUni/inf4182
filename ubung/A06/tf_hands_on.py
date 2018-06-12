import tensorflow as tf
sess = tf.Session()

#--------------------------------- Instructions-------------------------------#
# - Use only tensorflow functions for calculations.
# - Please clearly indicate the part number while printing the results.
# - When using random tensors as inputs, please also print the inputs as well.
# - Part (a) is already done for your reference.
#-----------------------------------------------------------------------------#

###############################################################################
# 1a (0 point): Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
x = tf.random_uniform([])
y = tf.random_uniform([])
result = tf.cond( x > y, lambda: tf.add(x,y), lambda: tf.subtract(x,y) )
print '\npart (a): ', sess.run([x,y,result])
###############################################################################

###############################################################################
# 1b (1 point): Create two 0-d tensors x and y from a normal distribution.
# Return x * y if x < y, x / y if x > y, x^2+y^2 otherwise.
# Hint: Look up tf.case().
###############################################################################

###############################################################################
# 1c (1 point): Create the tensor x of the value [[1, -2, -1], [0, 1, 2]] 
# and y as a tensor of ones with the same shape as x.
# Return a boolean tensor that yields Trues if absolute value of x equals 
# y element-wise.
# Hint: Look up tf.equal().
###############################################################################

###############################################################################
# 1d (1 point): Create a tensor x having 10 elements with random uniform numbers
# between -1 and 1 
# Get the indices of elements in x which are postive.
# Hint: Use tf.where().
# Then extract elements whose values are positive.
# Hint: Use tf.gather().
###############################################################################

###############################################################################
# 1e (2 point): Create two tensors x and y of shape 5 from any distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: Look up in TF documentation for methods to compute mean and sum
###############################################################################

###############################################################################
# 1f (2 pt): Create two random 2-d tensors x and y both of size 3 x 2.  
# - Concatenate x and y in axis 0  if the sum of all elements of x is greater 
#   than the sum of all elements of y
# - Otherwise, Concatenate x and y in axis 1 
# Hint: Use tf.concat()
###############################################################################

###############################################################################
# 1g (3 points): We want to find the pseudo inverse of a matrix A 
# Create a 3x3 tensor A = [[1,2,3],[3,6,3],[7,8,9]]
# Find the transpose of A (Atrans)
# Calculate the matrix B = (Atrans x A)
# Take the inverse of B (Binv)
# Compute the pseudo inverse matrix A_pinv = Binv x Atrans
# Find the inverse of A (A_inv) and print both A_inv and A_pinv
###############################################################################

