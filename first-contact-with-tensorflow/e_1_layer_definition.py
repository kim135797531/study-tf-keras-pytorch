# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 5장. 다중 계층 신경망(MNIST 데이터셋 - 숫자 분류 - 합성곱 신경망)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Layer(object):
    """신경망 계층 추상클래스를 정의

    execute 메소드를 실행하면, 신경망이 구성됨
    실제 학습은 런타임에 텐서플로우가 수행

    Attributes:
        data: 신경망 텐서
        :type data: tf.python.framework.ops.Tensor
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.data = None

    def relu(self):
        self.data = tf.nn.relu(self.data)

    def softmax(self):
        self.data = tf.nn.softmax(self.data)

    def get(self):
        """신경망을 통과한 출력 텐서 반환

        Returns:
            출력 텐서
            :rtype : tf.python.framework.ops.Tensor
        """
        return self.data

    @abstractmethod
    def execute(self, input_data):
        return self


class ConvolutionLayer(Layer):
    """합성곱 계층 클래스를 정의

    - 하나의 합성곱은 여러개의 커널을 포함함
    - 각각의 커널은 공유 행렬 W와 편향 b로 정의됨
    - 따라서, 하나의 커널은 이미지에서 한 종류의 특징만을 감지함 (W가 같으므로)
    - 따라서, 하나의 커널은 하나의 특징 맵을 만든다고 볼 수 있음

    Attributes:
        x: 윈도우 너비
        y: 윈도우 높이
        in_channels: 입력 채널 크기
        out_channels: 출력 채널 크기
        W: 가중치 텐서 W
        b: 편향 b
        :type x: int
        :type y: int
        :type in_channels: int
        :type out_channels: int
        :type W: tf.python.framework.ops.Tensor
        :type b: tf.python.framework.ops.Tensor
    """

    def __init__(self, shape, channel):
        """초기화

        Args:
            shape: 윈도우 크기
            channel: 입력 및 출력 채널 크기
            :type shape: list[int, int]
            :type channel: list[int, int]
        """
        super(self.__class__, self).__init__()
        self.x, self.y = shape
        self.in_channels, self.out_channels = channel
        self.W = tf.Variable(tf.truncated_normal([self.x, self.y, self.in_channels, self.out_channels], stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[self.out_channels]))

    def __convolution(self, input_data):
        """컨볼루션 수행

        strides는 한번에 몇 칸씩 이동할건지 지정
        padding은 채울 테두리의 크기 지정
        -> SAME은 원본과 같게, VALID는 유효 영역 안에만

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        """
        self.data = tf.nn.conv2d(input_data, self.W, strides=[1, 1, 1, 1], padding='SAME') + self.b

    def execute(self, input_data):
        """신경망 구성

        컨볼루션 실행후, ReLU 함수를 실행하는 신경망을 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : ConvolutionLayer
        """
        self.__convolution(input_data)
        self.relu()
        return self


class PoolingLayer(Layer):
    """맥스 풀링 계층 클래스를 정의

    - 하나의 특징은 보통 이미지의 여러 곳에 나타나는데,
      이 때 특징의 정확한 위치보다 다른 특징들과의 상대적 위치가 더 중요함
    - 쉽게 말하면 모두 같은 방법으로 이미지 정보를 깔아 뭉갰을 경우,
      어차피 다른 특징들과의 상대적 위치는 변하지 않기 때문에 여전히 특징 정보가 유효함

    Attributes:
        x: 풀링 너비
        y: 풀링 높이
        :type x: int
        :type y: int
    """

    def __init__(self, shape):
        """초기화

        Args:
            shape: 풀링 크기
            :type shape: list[int, int]
        """
        super(self.__class__, self).__init__()
        self.x, self.y = shape

    def __max_pool(self, input_data):
        """맥스 풀링 수행

        ksize는 맥스 풀링할 영역
        strides는 한번에 몇 칸씩 이동할건지
        padding은 채울 테두리의 크기 지정
        -> SAME은 원본과 같게, VALID는 유효 영역 안에만

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        """
        self.data = tf.nn.max_pool(input_data, ksize=[1, self.x, self.y, 1], strides=[1, self.x, self.y, 1], padding='SAME')

    def execute(self, input_data):
        """신경망 구성

        맥스 풀링을 실행하는 신경망을 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : ConvolutionLayer
        """
        self.__max_pool(input_data)
        return self


class HiddenLayer(Layer):
    """은닉 계층 클래스를 정의

    - 하나의 은닉 계층 클래스는 합성곱 연산 후 맥스 풀링까지 하는 것이 보통
    - 합성곱 계층과 맥스 풀링 계층을 하나로 묶음

    Attributes:
        conv_layer: 합성곱 계층
        pool_layer: 맥스 풀링 계층
        :type conv_layer: ConvolutionLayer
        :type pool_layer: PoolingLayer
    """

    def __init__(self, convolution_shape, channel, pooling_shape):
        """초기화

        Args:
            convolution_shape: 윈도우 크기
            channel: 입력 및 출력 채널 크기
            pooling_shape: 풀링 크기
            :type convolution_shape: list[int, int]
            :type channel: list[int, int]
            :type pooling_shape: list[int, int]
        """
        super(self.__class__, self).__init__()
        self.conv_layer = ConvolutionLayer(convolution_shape, channel)
        self.pool_layer = PoolingLayer(pooling_shape)

    def execute(self, input_data):
        """신경망 구성

        합성곱 계층과 맥스 풀링 계층을 연결하여 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : HiddenLayer
        """
        conv_data = self.conv_layer.execute(input_data).get()
        self.data = self.pool_layer.execute(conv_data).get()
        return self

    def get_flatten_size(self):
        """직렬화된 텐서의 크기 반환

        Returns:
            직렬화된 텐서의 크기
            :rtype : int
        """
        # TODO: 깔끔한 계산 방법 생각 or 아예 재설계
        return self.get().get_shape()[1].value * \
               self.get().get_shape()[2].value * \
               self.get().get_shape()[3].value

    def get_flatten(self):
        """신경망을 통과한 출력 텐서를 직렬화 해서 반환

        Returns:
            출력 텐서
            :rtype : tf.python.framework.ops.Tensor
        """
        return tf.reshape(self.get(), [-1, self.get_flatten_size()])


class FullyConnectedLayer(Layer):
    """완전 연결 계층 클래스를 정의

    - 기존 계층의 데이터를 취합

    Attributes:
        activate_function: 활성화 함수의 종류(relu, softmax 가능)
        in_channels: 입력 채널 크기
        out_channels: 출력 채널 크기
        W: 가중치 텐서 W
        b: 편향 b
        :type activate_function: int
        :type in_channels: int
        :type out_channels: int
        :type W: tf.python.framework.ops.Tensor
        :type b: tf.python.framework.ops.Tensor
    """

    RELU = 1000
    SOFTMAX = 1001

    def __init__(self, activate_function, channel):
        """초기화

        Args:
            activate_function: 윈도우 크기
            channel: 입력 및 출력 채널 크기
            :type activate_function: list[int, int]
            :type channel: list[int, int]
        """
        super(self.__class__, self).__init__()
        self.activate_function = activate_function
        self.in_channels, self.out_channels = channel
        self.W = tf.Variable(tf.truncated_normal([self.in_channels, self.out_channels], stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[self.out_channels]))

    def __matmul(self, input_data):
        """행렬 곱셈 수행

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        """
        self.data = tf.matmul(input_data, self.W) + self.b

    def __execute_relu(self, input_data):
        """신경망 구성

        가중치 텐서를 곱하고 ReLU 함수를 실행하는 신경망 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : FullyConnectedLayer
        """
        self.__matmul(input_data)
        self.relu()
        return self

    def __execute_softmax(self, input_data):
        """신경망 구성

        가중치 텐서를 곱하고 소프트맥수 함수를 실행하는 신경망 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : FullyConnectedLayer
        """
        self.__matmul(input_data)
        self.softmax()
        return self

    def execute(self, input_data):
        """신경망 구성

        활성화 함수를 선택하여 신경망 구성

        Args:
            input_data: 입력 텐서
            :param input_data: tf.python.framework.ops.Tensor
        Returns:
            구성된 신경망 인스턴스
            :rtype : FullyConnectedLayer
        """
        if self.activate_function == self.RELU:
            return self.__execute_relu(input_data)
        elif self.activate_function == self.SOFTMAX:
            return self.__execute_softmax(input_data)
        else:
            return self
