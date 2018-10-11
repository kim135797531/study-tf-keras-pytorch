import tensorflow as tf


def inference(x, keep_prob, n_in, n_hiddens, n_out):
    # 모델을 정의한다
    # ex) 은닉층은 몇 개이고, 각각 층은 몇 차원인가
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    # 입력층 - 은닉층, 은닉층 - 은닉층
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = x
            input_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i-1]

        W = weight_variable([input_dim, n_hidden])
        b = bias_variable([n_hidden])
        h = tf.nn.relu(tf.matmul(input, W) + b)
        output = tf.nn.dropout(h, keep_prob)

    # 은닉층 - 출력층
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)

    return y


def loss(y, t):
    # 오차 함수를 정의한다
    # ex) 교차 엔트로피 등
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
    return cross_entropy


def training(loss):
    # 학습 알고리즘을 정의한다
    # ex) 경사 하강법 등
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step


if __name__ == '__main__':
    # 1. 데이터를 준비한다
    # 2. 모델을 설정한다
    n_in, n_hiddens, n_out = 784, [200, 200, 200], 10
    keep_prob = tf.placeholder(tf.float32)

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    y = inference(x, keep_prob, n_in, n_hiddens, n_out)

    loss = loss(y, t)
    train_step = training(loss)
    # 3. 모델을 학습시킨다
    # 4. 모델을 평가한다
