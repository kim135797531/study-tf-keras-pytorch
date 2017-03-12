# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 5장. 다중 계층 신경망(MNIST 데이터셋 - 숫자 분류 - 합성곱 신경망)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

from tensorflow.examples.tutorials.mnist import input_data
from e_1_layer_definition import *

############################
# 0. 실험 파라미터 설정
############################
# NUM_IMAGES = 데이터셋의 크기(랜덤하게 뽑아 볼 이미지의 개수)
# NUM_LEARNS = 학습 횟수
# LEARN_SPEED = 학습 속도(ADAM 알고리즘에서의 학습 속도)
# KEEP_PROB = ReLU 함수를 적용하고 드롭아웃할 뉴런의 비율
NUM_IMAGES = 100
NUM_LEARNS = 1000
LEARN_SPEED = 1e-4
KEEP_PROB = tf.placeholder("float")

# NUM_KERNEL_1, NUM_KERNEL_2 = 합성곱 계층에서의 커널 개수
NUM_KERNEL_1 = 32
NUM_KERNEL_2 = 64

# WINDOW_WIDTH, WINDOW_HEIGHT = 합성곱 계층에서의 윈도우 크기(하나의 뉴런이 훑는 크기)
# POOLING_WIDTH, POOLING_HEIGHT = 풀링 계층에서의 풀링 크기
WINDOW_WIDTH = 5
WINDOW_HEIGHT = 5
POOLING_WIDTH = 2
POOLING_HEIGHT = 2

# NUM_CLASS = 완전 연결 계층에서의 ReLU 함수로 분류할 클래스의 크기
NUM_CLASS = 1024

############################
# 1. MNIST 데이터셋 다운로드
############################
# 텐서플로우 예제에서 제공되는 MNIST 데이터셋을 하위 폴더(MNIST_data)에 저장
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

############################
# 2. 텐서플로우 알고리즘 선언
############################
# 테스트 이미지셋 텐서를 정의. 임의의 이미지(None)들이 784개의 픽셀을 갖고 있음
x = tf.placeholder("float", [None, 784])
# 정답 레이블 텐서를 정의. 임의의 이미지(None)들이 레이블(0 ~ 9)을 갖고 있음
y_ = tf.placeholder("float", [None, 10])

# 차원 정보가 없던 텐서를, 1차원 텐서로 변환 (?, 784) -> (?, 28, 28, 1)
# -> 그냥 28x28 텐서를 이용하지 않고 굳이 28x28x1 텐서로 만들어 주는 이유는,
# tf.nn.conv2d(컨볼루션 연산)에 차원 정보가 필요하기 때문.
# -> 우리는 입력 이미지가 흑백 이미지임을 알고 있지만, 위의 컨볼루션 연산 함수를 사용할 때
# 명시적으로 입력 이미지가 1차원(흑백)임을 전달해야 함
# -> (책에 왜 1차원을 추가해야 하는지 설명이 나와있지 않음)
# [batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

############################
# 2-a. 은닉 계층 선언
# -> 합성곱 계층 및 맥스 풀링 계층을 포함하고 있음
############################
# 은닉 계층 1
# 5x5 사이즈 픽셀에 대해, 1채널의 가중치 행렬을, 32개 저장
# (?, 28, 28, 1) -> (?, 14, 14, 32)
hidden_layer1 = HiddenLayer(
    [WINDOW_WIDTH, WINDOW_HEIGHT],
    [1, NUM_KERNEL_1],
    [POOLING_WIDTH, POOLING_HEIGHT]).execute(x_image)

# 은닉 계층 2
# (?, 14, 14, 32) -> (?, 7, 7, 64) ->
# (?, 7 * 7 * 64) (다음 단계에서 직렬화된 텐서가 필요)
hidden_layer2 = HiddenLayer(
    [WINDOW_WIDTH, WINDOW_HEIGHT],
    [NUM_KERNEL_1, NUM_KERNEL_2],
    [POOLING_WIDTH, POOLING_HEIGHT]).execute(hidden_layer1.get())

############################
# 2-b. 완전 연결 계층 선언
# -> 은닉 계층에서 만들어진 특징 맵들을 모두 취합하여 클래스로 분류함
############################
# 완전 연결 계층 1 (?, 7 * 7 * 64) -> (?, 1024)
# -> 은닉 계층의 데이터를 취합하여, 활성화 함수로 ReLU 함수 적용
# -> ReLU 함수 = Max(0, x) = 양수일 때만 활성화
# -> 원래 바로 시그모이드 함수(소프트맥스)를 적용해도 되나,
#    학습 속도(미분값)가 너무 느려지기(작아지기) 때문에 ReLU 함수를 한 번 거침
# -> (책에 왜 ReLU 계층이 필요한지 설명이 나와있지 않음)
fully_connected_layer1 = FullyConnectedLayer(
    FullyConnectedLayer.RELU,
    [hidden_layer2.get_flatten_size(), NUM_CLASS]).execute(hidden_layer2.get_flatten())

# 모든 클래스(뉴런)을 다 사용하면 정확도는 높아지나, 오버피팅 가능성이 있음
# 랜덤으로 클래스를 제거(드롭아웃)하여 오버피팅 확률을 낮춤
fully_connected_layer1_drop = tf.nn.dropout(fully_connected_layer1.get(), KEEP_PROB)

# 완전 연결 계층 2 (?, 1024) -> (?, 10)
# 신경망 학습. 출력층에 소프트맥스 함수를 이용함
fully_connected_layer2 = FullyConnectedLayer(
    FullyConnectedLayer.SOFTMAX,
    [NUM_CLASS, 10]).execute(fully_connected_layer1_drop)

############################
# 2-c. 만들어진 신경망의 학습법 정의
############################
# 다중 계층 신경망의 최종 출력을 y로 정의
y = fully_connected_layer2.get()

# 교차 엔트로피 - 신경망 학습에서 주로 쓰이는 함수
# y'는 실제 분포, y는 예측 분포일 때,
# 어떤 현상을 y를 이용해 특정하기 필요한 비트 수의 평균치
# y' = y 일때(여기서는 숫자 분류 결과가 정답과 같을 때) 최솟값을 가짐
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

############################
# ADAM 알고리즘
# -> 발견된 지 얼마 되지 않아(2014년 논문) 아직 쉽게 설명된 곳이 없음
# -> 텐서플로우 API 문서도 논문 링크만 달랑 있음(http://arxiv.org/pdf/1412.6980.pdf)
# ADAM 알고리즘의 간단한 설명
# -> http://aikorea.org/cs231n/neural-networks-3
# - Adagrad 알고리즘
# --> 시간에 따라 그레디언트 제곱값을 추적해 학습률을 조정
# --> cache += dx**2
# --> x += - learning_rate * dx / (np.sqrt(cache) + eps)
# - RMSprop 알고리즘
# --> Adagrad 알고리즘 + 이동평균 사용(감쇠향 추가)
# --> cache = decay_rate * cache + (1 - decay_rate) * dx**2
# --> x += - learning_rate * dx / (np.sqrt(cache) + eps)
# - ADAM 알고리즘
# --> RMSprop 알고리즘 + 모멘텀 혼합
# --> m = beta1*m + (1-beta1)*dx
# --> v = beta2*v + (1-beta2)*(dx**2)
# --> x += - learning_rate * m / (np.sqrt(v) + eps)
############################
# ADAM 알고리즘을 정의
# 학습 속도 - 1e-4
# 교차 엔트로피를 비용 함수로 사용해, Adam 알고리즘을 실행
train_step = tf.train.AdamOptimizer(LEARN_SPEED).minimize(cross_entropy)

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
writer = tf.summary.FileWriter("./log/e_multi_layer_neural_networks", sess.graph)

# 변수들 초기화
init = tf.global_variables_initializer()
sess.run(init)

# 1000번 반복 학습
for i in range(NUM_LEARNS):
    # 각 반복마다 임의의 숫자 100개를 뽑아 학습시킴
    batch_xs, batch_ys = mnist.train.next_batch(NUM_IMAGES)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, KEEP_PROB: 0.5})

    # 테스트 실행
    merged_summary_value, accuracy_value = \
        sess.run(
            [merged_summary, accuracy],
            feed_dict={x: mnist.test.images, y_: mnist.test.labels, KEEP_PROB: 1.0}
        )

    # 텐서보드에 결과 출력
    writer.add_summary(merged_summary_value, i)
    print("testing accuracy %g" % accuracy_value)

    # 100번에 한 번씩 신경망과 학습 데이터 사이의 정답률 출력
    if i % 100 == 0:
        train_accuracy = sess.run(
            accuracy,
            feed_dict={x: batch_xs, y_: batch_ys, KEEP_PROB: 1.0}
        )
        print("training accuracy %g (step %d)" % (i, train_accuracy))
