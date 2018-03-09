import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

# 1. 모델을 정의한다.
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 2. 오차 함수를 정의한다.
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 3. 최적화 기법을 정의한다.
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 4. 세션을 초기화한다.
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 5. 학습한다.
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

classified = correct_prediction.eval(session=sess, feed_dict = {
    x: X,
    t: Y
})

print('classified:')
print(classified)
print()

prob = y.eval(session=sess, feed_dict={
    x: X,
    t: Y
})

print('output probability:')
print(prob)

print('w:', sess.run(w))
print('b:', sess.run(b))
