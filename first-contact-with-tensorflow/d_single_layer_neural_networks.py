# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 4장. 단일 계층 신경망(MNIST 데이터셋 - 숫자 분류)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

############################
# 0. 실험 파라미터 설정
############################
# NUM_IMAGES = 데이터셋의 크기(랜덤하게 뽑아 볼 이미지의 개수)
# NUM_LEARNS = 학습 횟수
# LEARN_SPEED = 학습 속도(경사 하강법에서의 학습 속도)
NUM_IMAGES = 100
NUM_LEARNS = 1000
LEARN_SPEED = 0.01

############################
# 1. MNIST 데이터셋 다운로드
############################
# 텐서플로우 예제에서 제공되는 MNIST 데이터셋을 하위 폴더(MNIST_data)에 저장
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

############################
# 2. 텐서플로우 알고리즘 선언
############################
# 가중치 텐서 W를 정의. 각 픽셀(784개)에 대해 레이블(0 ~ 9)마다의 가중치를 저장 (기울기)
# 편향 텐서 b를 정의. 각 레이블(0 ~ 10)마다의 편향을 저장 (y절편)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 테스트 이미지셋 텐서를 정의. 임의의 이미지(None)들이 784개의 픽셀을 갖고 있음
x = tf.placeholder("float", [None, 784])

# 단일 신경망 학습. 출력층에 소프트맥스 함수를 이용함 (입력층 -> 출력층)
# 각 이미지에 가중치를 곱하고 편향을 더함 (?, 784) -> (?, 10)
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 정답 레이블 텐서를 정의. 임의의 이미지(None)들이 레이블(0 ~ 9)을 갖고 있음
y_ = tf.placeholder("float", [None, 10])

# 교차 엔트로피 - 신경망 학습에서 주로 쓰이는 함수
# y'는 실제 분포, y는 예측 분포일 때,
# 어떤 현상을 y를 이용해 특정하기 필요한 비트 수의 평균치
# y' = y 일때(여기서는 숫자 분류 결과가 정답과 같을 때) 최솟값을 가짐
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 경사 하강법 알고리즘을 정의
# 학습 속도 - 0.01
# 교차 엔트로피를 비용 함수로 사용해, 경사 하강법을 실행
train_step = tf.train.GradientDescentOptimizer(LEARN_SPEED).minimize(cross_entropy)

# 학습을 마친 후 실행시킬 테스팅 정의
# 각 이미지 별 레이블 인덱스를 구하고(argmax), 정답 레이블과 같으면 True로 채워 넣음
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# True, False 로 이루어진 텐서를 깔아 뭉개면서, 평균을 남김(정확도) (?) -> ()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

############################
# 3. 텐서보드 표시 선언
############################
tf.summary.scalar("cross entropy", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

############################
# 4. 텐서플로우 알고리즘 실행
############################
sess = tf.Session()
writer = tf.summary.FileWriter("./log/d_single_layer_neural_networks", sess.graph)

# 변수들 초기화
init = tf.global_variables_initializer()
sess.run(init)

# 1000번 반복 학습
for i in range(NUM_LEARNS):
    # 각 반복마다 임의의 숫자 100개를 뽑아 학습시킴
    batch_xs, batch_ys = mnist.train.next_batch(NUM_IMAGES)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 테스트 실행
    merged_summary_value, accuracy_value = \
        sess.run(
            [merged_summary, accuracy],
            feed_dict={x: mnist.test.images, y_: mnist.test.labels}
        )

    # 텐서보드에 결과 출력
    writer.add_summary(merged_summary_value, i)
    print(accuracy_value)

