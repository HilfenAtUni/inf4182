import numpy as np

#
# (b)
#
def softmax(a):
    """compute the softmax for the preactivations a.
    a can eithe be a vector or a matrix. If a is a matrix, calculate the
    softmax over each row"""
    #TODO

#
# (c)
#
def softmax_crossentropy(a, y):
    """compute the softmax-cossentropy between the preactivations a and the
    correct class y.
    y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
    a and y can eithe be a vector and an int or a matrix and a vector. If a is
    a matrix, calculate the softmax-crossentropy of each row with the
    corresponding element of the vector"""
    #TODO

#
# (d)
#
def grad_softmax_crossentropy(a, y):
    """compute the gradient of the softmax-cossentropy between the
    preactivations a and the correct class y with respect to the preactivations
    a.
    y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
    a and y can eithe be a vector and an int or a matrix and a vector. If a is
    a matrix, calculate the gradient of the softmax-crossentropy of each row
    with the corresponding element of the vector"""
    #TODO

#
# (e)
#

# To compute the numerical gradient at a point (a,y), for component i compute
# '(ce(a+da,y)-ce(a,y))/e' where 'da[i] = e' and the other entries of 'da' are
# zero and e is a small number, e.g. 0.0001 (i.e. use the finite differences
# method for each component of the gradient separately).

# implemented correctly, the difference between analytical and numerical
# gradient should be of the same magnitude as e
