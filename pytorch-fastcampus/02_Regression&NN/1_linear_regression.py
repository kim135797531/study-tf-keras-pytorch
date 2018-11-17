# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/02_Regression&NN/0_Linear_code/1_linear_regression.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#
# 선형 회귀 예제
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from visdom import Visdom

Visdom().delete_env('main')
viz = Visdom()

print("\n1. 데이터 생성")
# y=2x+3 꼴 그래프를 만드는데, 노이즈를 조금 준다.
num_data = 1000
num_epoch = 1000

noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)
x = init.uniform_(torch.FloatTensor(num_data, 1), -10, 10)
y = 2*x + 3
y_noise = (2*x + 3) + noise

input_data = torch.cat([x, y_noise], dim=1)
print(input_data)

opts = dict(
    xtickmin=-10,
    xtickmax=10,
    xtickstep=1,
    ytickmin=-20,
    ytickmax=20,
    ytickstep=1,
    markersymbol='dot',
    markersize=5,
    markercolor=np.random.randint(0, 255, num_data)
)

win = viz.scatter(X=input_data, opts=opts)

print("\n2. 모델과 근사 방법을 정의")
model = nn.Linear(1, 1)  # 입력 1개, 출력 1개
output = model(x)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("\n3. 훈련")
loss_arr = []
for i in range(num_epoch):
    optimizer.zero_grad()  # 그라디언트 초기화
    output = model(x)

    loss = loss_func(output, y_noise)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss)

    loss_arr.append(loss.data.numpy().item(0))

print("\n4. 학습된 파라미터 확인")
# 제대로 2x+3을 학습했는지 확인
param_list = list(model.parameters())
print(param_list[0].data, param_list[1].data)

print("\n5. 결과 시각화")
viz.scatter(X=torch.cat([x, output], dim=1), opts=opts)

print("\n6. Loss 시각화")
x = [i for i in range(num_epoch)]
viz.line(X=x, Y=loss_arr)
