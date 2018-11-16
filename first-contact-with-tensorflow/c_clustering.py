# coding=utf-8
#
# 텐서플로 첫걸음(조르디 토레스 지음, 박해선 옮김) 공부용 코드
# 3장. 군집화(K-평균 알고리즘)
#
# https://github.com/jorditorresBCN/FirstContactWithTensorFlow
# https://github.com/rickiepark/first-steps-with-tensorflow
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

############################
# 0. 실험 파라미터 설정
############################
# NUM_POINTS = 데이터셋의 크기(분류할 점의 개수)
# NUM_LEARNS = 학습 횟수
# NUM_K = K-평균 알고리즘에서의 K값(중심점의 개수)
NUM_POINTS = 3000
NUM_LEARNS = 500
NUM_K = 4

############################
# 1. 랜덤한 데이터셋 생성
############################
num_points = NUM_POINTS
vectors_set = []

# [[0.1, 0.1], [3.0, 1.0], ... , [3.0, 1.0]]
for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])

############################
# 2. 텐서플로우 알고리즘 선언
############################
# 뽑을 중심점의 개수
k = NUM_K

# Python 리스트를 텐서 형태로 표현 (2000)
vectors = tf.constant(vectors_set)

# 초기 중심값은 무작위로 k개를 뽑아 정함 (k, 2)
# 1. vectors 배열을 무작위로 섞음
# 2. D0차원의 맨 앞(0)에서부터 (k)개를 뽑음
# 3. D1차원의 맨 앞(0)에서부터 맨 뒤까지(-1)를 뽑아서 D0차원에 채워 넣음
centroids = tf.Variable(
                tf.slice(
                    tf.random_shuffle(vectors),
                    [0, 0],
                    [k, -1]
                )
            )

# 중심과의 거리를 구하는 연산을 k개의 점에 대해 수행해야 하므로, 먼저 차원을 확장함
# vectors = (2000, 2)차원의 텐서 -> 차원을 추가하여, (1, 2000, 2)차원의 텐서로 만듦
# centroids = (k, 2)차원의 텐서 -> 차원을 추가하여, (k, 1, 2)차원의 텐서로 만듦
# 크기가 1인 차원은, 추후의 텐서 연산 시 다른 텐서의 해당 차원 크기에 맞게 확장되어 계산을 반복하게 됨
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

# 중심과의 거리를 구함 (k, 2000, 2) -> (k, 2000)
# 1. tf.subtract 함수가 두 텐서의 원소를 뺌. D2차원의 크기가 2로 같으므로 D2차원에서 뺄셈이 이루어짐 (k, 2000, 2)
# 2. tf.square 함수가 각 원소를 제곱함 (xdiff^2, ydiff^2), (k, 2000, 2)
# 3. tf.reduce_sum 함수가 D2차원의 모든 원소를 더하면서 D2차원을 없앰 (k, 2000)
# 결론은 (k개의 중심점과 2000개의 점 사이의 거리)들을 표시하는 텐서가 남음
# for 문으로 전부 순회하며 빼고 제곱할 것을, 함수 하나로 처리함
distances = \
    tf.reduce_sum(
        tf.square(tf.subtract(expanded_vectors, expanded_centroids)),
        2
    )

# 각 지점별로 가장 가까운 중심을 구함 (k, 2000) -> (2000)
# tf.argmin 함수는 지정된 차원을 깔아 뭉개면서, 그 차원이 갖고 있던 값 중 *최솟값*의 *인덱스*만 남긴다고 이해하면 쉬움
# tf.reduce_min 함수는 지정된 차원을 깔아 뭉개면서, 그 차원이 갖고 있던 값 중 *최솟값*을 남기므로 조금 다름
# 여기서는 D0 차원(중심점들과의 거리)을 없애면서, 가장 가까운 중심점의 인덱스를 남김
# for 문으로 전부 순회하며 최솟값을 검사할 것을, 함수 하나로 처리함
assignments = tf.argmin(distances, 0)

# 각 군집별로 새로운 중심을 구함 (k, 2)
# 원 소스는 concat을 이용해서 재귀형으로 구현되었지만, 공부를 위해서 절차형으로 바꿔 보았음
#
# 1. tf.equal 함수를 이용해 주어진 텐서 중에서 특정 중심점(현재 변수 c로 순회중)이 가까운 군집을 만듦 (2000)
# 2. tf.where 함수를 이용해 True를 갖고 있는 인덱스의 텐서를 만듦 (2000, 1)
# 3. tf.squeeze 함수를 이용해 필요없는 차원을 없앰 (2000, 1) -> (2000)
#    -> [[0], [2], ..., [1970]] -> [0, 2, ..., 1970]
# 4. tf.gather 함수는 indices 텐서(인덱스 정보)를 params 텐서에서 가져온 실제 데이터로 채운다고 이해하면 쉬움 (2000) -> (2000, 2)
#    여기서 indices 텐서는 [0, 2, ..., 1970] (2000)꼴이므로,
#    결과는 [[0.1, 0.1], ..., [3.0, 1.0]] (2000, 2)꼴이 됨
# 5. tf.reduce_min 함수는 지정된 차원을 깔아 뭉개면서, 그 차원이 갖고 있던 값들의 *평균*을 남김 (2000, 2) -> (2)
#    -> [[0.1, 0.1], ..., [3.0, 1.0]] -> [7.7, 7.7] (새로운 중심점)
# 참고로 if 문으로 텐서를 순회하며 군집을 찾을 것을, equal 과 where 함수로 처리함
new_centroid_set = []

for c in range(k):
    cluster_indexes = tf.squeeze(tf.where(tf.equal(assignments, c)))
    cluster_vectors = tf.gather(vectors, cluster_indexes)
    reduced_mean = tf.reduce_mean(cluster_vectors, 0)
    new_centroid_set.append(reduced_mean)

# k개의 새로운 중심점 리스트를 텐서로 변환 (k, 2)
new_centroids = tf.convert_to_tensor(new_centroid_set)

# 새로운 중심을 구하는 원본 코드 백업
# 재귀형으로 구현되어 깔끔하고 속도가 빠를 것이나, 가독성이 떨어짐
# new_centroids = \
#     tf.concat([
#         tf.reduce_mean(
#             tf.gather(
#                 vectors,
#                 tf.reshape(
#                     tf.where(tf.equal(assignments, c)),
#                     [1, -1]
#                 )
#             ),
#             1
#         ) for c in xrange(k)
#         ], 0)

# 기존 중심점들 텐서를 새롭게 구한 중심점들로 업데이트
update_centroids = tf.assign(centroids, new_centroids)

############################
# 3. 텐서플로우 알고리즘 실행
############################
sess = tf.Session()

# 변수들 초기화
init = tf.global_variables_initializer()
sess.run(init)

# 100번 반복 학습
# 원 소스는 아래와 같이 매 스텝마다 3개를 노드를 실행하지만,
#
# for step in xrange(100):
#     _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])
#
# centroids 노드와 assignments 노드는 update_centroids 의 하위 노드로서 어차피 실행되므로 굳이 실행할 필요가 없음
# 학습 도중 값을 보고 싶다면 실행하고, 아니면 아래 소스처럼 학습이 끝난 후 한 번 호출해서 값만 확인하면 됨
for step in range(NUM_LEARNS):
    sess.run(update_centroids)

centroid_values, assignment_values = sess.run([centroids, assignments])

############################
# 4. 데이터셋 그래프로 출력 해 보기
############################
# 데이터 dict 정의
data = {"x": [], "y": [], "cluster": []}

# dict의 각 항목에 데이터 삽입
for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

# 데이터를 가지고 데이터 프레임 생성
df = pd.DataFrame(data)

# 화면에 표시될 방법 정의
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)

# 화면에 표시
plt.show()
