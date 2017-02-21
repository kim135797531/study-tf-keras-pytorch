# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 1장. 텐서플로 기본 다지기
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

import tensorflow as tf

############################
# 1. 텐서플로우 알고리즘 선언
############################
a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

############################
# 2. 텐서플로우 알고리즘 실행
############################
sess = tf.Session()

print sess.run(y, feed_dict={a: 3, b: 3})
