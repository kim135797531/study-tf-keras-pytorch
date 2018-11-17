# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/02_Regression&NN/1_NN_code/Spiral_Multiclass.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#
# 단순 역전파 신경망으로 나선 형태 다분류 학습
# GTX1060 OC 6GB 기준 실행 시간 14.74초
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
num_data = 100
num_epoch = 10000
dimensionality = 2
classes = 3

x = np.zeros((num_data*classes, dimensionality))
y = np.zeros(num_data*classes, dtype='uint8')

for j in range(classes):
    ix = range(num_data*j, num_data*(j+1))
    r = np.linspace(0.0, 1, num_data)
    t = np.linspace(j*4, (j+1)*4, num_data) + np.random.randn(num_data)*0.2
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j+1  # 라벨 1~3

viz.scatter(X=x, Y=y)

x = torch.from_numpy(x).type_as(torch.FloatTensor())
y_ = torch.from_numpy(y-1).type_as(torch.LongTensor())  # 라벨 0~2
print(x.size(), y_.size())

print("\n2. 모델과 근사 방법을 정의")
# 2->20->10->5->5->3
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 3)
).cuda()

output = model(x.cuda())
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n3. 훈련")
label = y_.cuda()
for i in range(num_epoch):
    optimizer.zero_grad()  # 그라디언트 초기화
    output = model(x.cuda())

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)

print("\n4. 학습된 파라미터 확인")
param_list = list(model.parameters())
print(param_list)

print("\n5. 결과 시각화")
viz.scatter(X=x.cpu(), Y=output.argmax(dim=1).cpu() + 1)

print("--- %s seconds ---" % (time.time() - start_time))