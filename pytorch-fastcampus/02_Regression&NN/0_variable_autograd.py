# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/02_Regression&NN/0_Linear_code/0_Variable_Autograd.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#
# Autograd 연습
#

import torch

print("\n1. 선언")
# 텐서 선언
x = torch.rand(3, 4)
print(x)

# 변수 선언, 그런데 기존 torch.autograd.Variable은 deprecated 되었다.
# 앞으로는 그냥 Tensor 쓰면 된다 해서 생략.
print(x.data)
print(x.grad)
print('기본 텐서의 requires_grad:', x.requires_grad)

# 그라디언트가 필요한 변수 선언
x = torch.rand(3, 4, requires_grad=True)
print('그라디언트 필요 텐서의 requires_grad:', x.requires_grad)

# 그라디언트 모드 켜기
torch.set_grad_enabled(True)
y = x * 2
print('그라디언트 모드 켰을 때 생긴 텐서의 requires_grad:', y.requires_grad)

# 그라디언트 계산이 필요하지 않은 변수 선언
torch.set_grad_enabled(False)
y = x * 2
print('그라디언트 모드 껐을 때 생긴 텐서의 requires_grad:', y.requires_grad)
torch.set_grad_enabled(True)

print("\n2. 그래프 생성과 역전파")
# tensor.backward() -> 리프노드로 오차 역전파 계산 (자동 그라디언트 계산)
x = torch.ones(3, 4, requires_grad=True)
y = x**2 + 4*x
z = 2*y + 3

criterion = torch.nn.MSELoss()
loss = criterion(z, torch.ones(3, 4))
print(loss)

loss.backward()
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy, 리프노드에만 그라디언트 적용됨
print(z.grad)  # dz/dz, 리프노드에만 그라디언트 적용됨
