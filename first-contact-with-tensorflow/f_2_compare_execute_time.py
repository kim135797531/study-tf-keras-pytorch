# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 6장. 병렬처리(A^n + B^n의 수행시간 비교 in 싱글 GPU 또는 멀티 GPU)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
# https://github.com/aymericdamien/TensorFlow-Examples
#

import datetime
import numpy as np
import tensorflow as tf

############################
# 0. 실험 파라미터 설정
############################
HAS_GPU = False
CPU = '/cpu:0'
GPU0 = '/gpu:0' if HAS_GPU else CPU
GPU1 = '/gpu:1' if HAS_GPU else CPU
GPU_LIST = [GPU0, GPU1]
N = 10
SIZE = 1000


############################
# 1. 텐서플로우 알고리즘 선언
############################
def matpow(M, n):
    return M if n < 1 else tf.matmul(M, matpow(M, n - 1))

A = np.random.rand(1000, 1000).astype('float32')
B = np.random.rand(1000, 1000).astype('float32')

c1 = []
c2 = []

with tf.device(GPU0):
    a = tf.constant(A)
    b = tf.constant(B)
    c1.append(matpow(a, N))
    c2.append(matpow(b, N))

with tf.device(CPU):
    sum_with_single_gpu = tf.add_n(c1)

with tf.device(GPU0):
    a = tf.constant(A)
    c2.append(matpow(a, N))

with tf.device(GPU1):
    b = tf.constant(B)
    c2.append(matpow(b, N))

with tf.device(CPU):
    sum_with_multi_gpu = tf.add_n(c2)


############################
# 2. 텐서플로우 알고리즘 실행
############################
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

single_start = datetime.datetime.now()
sess.run(sum_with_single_gpu)
single_end = datetime.datetime.now()

multi_start = datetime.datetime.now()
sess.run(sum_with_multi_gpu)
multi_end = datetime.datetime.now()

print("Single GPU computation time: " + str(single_end - single_start))
print("Multi GPU computation time: " + str(multi_end - multi_start))
