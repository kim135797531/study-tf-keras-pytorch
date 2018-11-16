# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 2장. 선형회귀분석
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

############################
# 0. 실험 파라미터 설정
############################
# NUM_POINTS = 데이터셋의 크기(추정할 점의 개수)
# NUM_LEARNS = 학습 횟수
# LEARN_SPEED = 학습 속도(경사 하강법에서의 학습 속도)
NUM_POINTS = 1000
NUM_LEARNS = 12
LEARN_SPEED = 0.5

############################
# 1. 랜덤한 데이터셋 생성
############################
num_points = NUM_POINTS
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

############################
# 2. 텐서플로우 알고리즘 선언
############################
# Variable 메소드 = 그래프 자료구조에서의 변수를 선언
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# 생성한 x_data에, W와 b를 연산 (선형함수)
y = W * x_data + b

# 비용 함수 - 생성된 y와 데이터 y_data값 사이의 거리를, 제곱한 것들의, 평균을 계산
loss = tf.reduce_mean(tf.square(y - y_data))

# 경사 하강법 알고리즘을 정의
# 학습 속도 - 0.5
optimizer = tf.train.GradientDescentOptimizer(LEARN_SPEED)

# 비용 함수 loss에 경사 하강법 알고리즘을 적용
train = optimizer.minimize(loss)

############################
# 3. 텐서플로우 알고리즘 실행
############################
sess = tf.Session()

# 변수들 초기화
init = tf.global_variables_initializer()
sess.run(init)

# 12번 반복 학습
for step in range(NUM_LEARNS):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))

# 학습된 W와 b 출력
print(sess.run(W), sess.run(b))

############################
# 4. 데이터셋 그래프로 출력 해 보기
############################
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlim(-2, 2)
plt.ylim(0.1, 0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
