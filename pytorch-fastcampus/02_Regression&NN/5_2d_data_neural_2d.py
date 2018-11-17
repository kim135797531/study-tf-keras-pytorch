# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/02_Regression&NN/1_NN_code/2d_data/neural_2d.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#
# 단순 역전파 신경망으로 우물형 그래프(다변수 방정식) 학습
# GTX1060 OC 6GB 기준 실행 시간 14.98초
#

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from visdom import Visdom

start_time = time.time()
Visdom().delete_env('main')
viz = Visdom()

print("\n1. 데이터 생성")
# z= x^2 + y^2 꼴 우물 그래프를 만드는데, 노이즈를 조금 준다.
num_data = 1000
num_epoch = 10000

x_noise = init.normal_(torch.FloatTensor(num_data, 1), std=0.1)
y_noise = init.normal_(torch.FloatTensor(num_data, 1), std=0.1)
x = init.uniform_(torch.FloatTensor(num_data, 1), -10, 10)
y = init.uniform_(torch.FloatTensor(num_data, 1), -10, 10)
z = x**2 + y**2
z_noise = x_noise**2 + y_noise**2

input_data = torch.cat([x, y, z_noise], dim=1)
print(input_data)

opts = dict(
    markersize=1,
    markercolor=np.ndarray(shape=[num_data, 3], dtype=float, buffer=[51, 153, 255]*np.ones(shape=[num_data, 3]))
)

win = viz.scatter(X=input_data, opts=opts)

print("\n2. 모델과 근사 방법을 정의")
# 2->20->10->5->5->1
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
).cuda()
x_y = torch.cat([x, y], dim=1).cuda()
loss_func = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n3. 훈련")
loss_arr = []
label = z_noise.cuda()
for i in range(num_epoch):
    optimizer.zero_grad()  # 그라디언트 초기화
    output = model(x_y)

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    loss_arr.append(loss.cpu().data.numpy().item(0))

    if i % 100 == 0 and i < 1000:
        print(loss)
        data = torch.cat([x_y.cpu(), output.cpu()], dim=1)

        opts = dict(
            markersize=1,
            markercolor=np.ndarray(shape=[num_data, 3], dtype=float,
                                   buffer=128 * np.ones(shape=[num_data, 3]))
        )
        win_2 = viz.scatter(X=data, opts=opts)


print("\n4. 학습된 파라미터 확인")
# 제대로 x^2 + y^2을 학습했는지 확인
param_list = list(model.parameters())
print(param_list)

print("\n5. 결과 시각화")
data = torch.cat([x_y.cpu(), output.cpu()], dim=1)
win_2 = viz.scatter(X=data, opts=opts)

print("\n6. Loss 시각화")
x = [i for i in range(num_epoch)]
viz.line(X=x, Y=loss_arr)

print("--- %s seconds ---" % (time.time() - start_time))