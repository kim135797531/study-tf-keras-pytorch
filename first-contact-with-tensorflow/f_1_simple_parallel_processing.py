# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 6장. 병렬처리(기본 사용법)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

import tensorflow as tf

############################
# 0. 실험 파라미터 설정
############################
HAS_GPU = False
CPU = '/cpu:0'
GPU0 = '/gpu:0' if HAS_GPU else CPU
GPU1 = '/gpu:1' if HAS_GPU else CPU
GPU_LIST = [GPU0, GPU1]

############################
# 1. 텐서플로우 알고리즘 선언
############################
c = []
for gpu in GPU_LIST:
    with tf.device(gpu):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c.append(tf.matmul(a, b))
    with tf.device(CPU):
        # 모든 텐서를 더함
        result = tf.add_n(c)

############################
# 2. 텐서플로우 알고리즘 실행
############################
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(result)
