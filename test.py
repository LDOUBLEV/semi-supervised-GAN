import tensorflow as tf
import numpy as np

# a = [[0], [1], [3], [2], [4]]
# b = tf.one_hot(a, 5)
# c = [0, 1, 2, 3, 4]
# with tf.Session() as sess:
#     d = tf.argmax(b)
#     print sess.run(d)
# for idx in range(10):
#     a = 0 if (idx+1)%2 else 1
#     print idx, ':', a

# alpha = np.ones((5, 1))*2
# b = np.ones((5, 4, 4, 1))*0.25
# b = np.reshape(b, (5, 4*4))
# print alpha*b
# global a
a = 0
b = np.random.uniform(0., 10, [10])
# b = np.random.
print b
for i in range(10):
    if b[i]>a:
        print 'b>a'
        a = b[i]
print 'the max num of b is {}'.format(a)

