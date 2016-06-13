## tensorflow logistic regression file 

import tensorflow as tf 

x = tf.placeholder(tf.float32, [None, 71553])
# if category then 71553, 12 
# if word by word then 71533, 60
W = tf.Variable(tf.zeros([71553, 12]))
b = tf.Variable(tf.zeros([12]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 12])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

f = open('some name')
X,Y = pickle.load(f)

for i in range(1000):
  #batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_x, batch_y = X,Y;
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: X, y_: Y}))
