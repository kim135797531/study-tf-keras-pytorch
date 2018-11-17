# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/02_Regression&NN/1_NN_code/1d_data/0_neural_quadratic.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#
# 단순 역전파 신경망으로 2차 방정식 학습
# GTX1060 OC 6GB 기준 실행 시간 5.46초
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
# y= 8x^2+7x+3 꼴 2차 방정식의 그래프를 만드는데, 노이즈를 조금 준다.
num_data = 1000
num_epoch = 5000

noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)
x = init.uniform_(torch.FloatTensor(num_data, 1), -15, 15)
y = 8*(x**2) + 7*x + 3
y_noise = y + noise

input_data = torch.cat([x, y_noise], dim=1)
print(input_data)

opts = dict(
    xtickmin=-15,
    xtickmax=15,
    xtickstep=1,
    ytickmin=0,
    ytickmax=500,
    ytickstep=1,
    markersymbol='dot',
    markersize=5,
    markercolor=np.random.randint(0, 255, num_data)
)

win = viz.scatter(X=input_data, opts=opts)

print("\n2. 모델과 근사 방법을 정의")
# 1->6->10->6->1
model = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
).cuda()
output = model(x.cuda())
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.0005)

print("\n3. 훈련")
loss_arr = []
label = y_noise.cuda()
for i in range(num_epoch):
    optimizer.zero_grad()  # 그라디언트 초기화
    output = model(x.cuda())

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)

    loss_arr.append(loss.cpu().data.numpy().item(0))

print("\n4. 학습된 파라미터 확인")
# 제대로 8x^2+7x+3을 학습했는지 확인
param_list = list(model.parameters())
print(param_list)

print("\n5. 결과 시각화")
# win_2 = viz.scatter(X=input_data, opts=opts)
# viz.scatter(X=torch.cat([x, output.cpu().data], dim=1), win=win_2, update='append', opts=opts)
viz.scatter(X=torch.cat([x, output.cpu().data], dim=1), opts=opts)

print("\n6. Loss 시각화")
x = [i for i in range(num_epoch)]
viz.line(X=x, Y=loss_arr)

print("--- %s seconds ---" % (time.time() - start_time))