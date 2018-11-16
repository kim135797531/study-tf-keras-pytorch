# coding=utf-8
#
# PyTorch로 시작하는 딥러닝 입문 CAMP 공부용 코드
# PyTorch-FastCampus/01_DL&Pytorch/2_pytorch_tensor_basic.ipynb
#
# https://github.com/GunhoChoi/PyTorch-FastCampus
#

import numpy as np
import torch
import torch.nn.init as init

print("\n1. 랜덤")
# torch.rand(행, 열) -> 균등분포 [0,1)
# torch.randn(행, 열) -> 정규분포 Z(0,1)
# torch.randperm(n) -> 순열 0~n
print(torch.rand(2, 3))
print(torch.randn(2, 3))
print(torch.randperm(5))

print("\n2. 텐서 생성")
# torch.zeros(행, 열)
# torch.ones(행, 열)
# torch.arange(시작, 끝, 간격) -> '간격'을 갖는 [시작, 끝)
print(torch.zeros(2, 3))
print(torch.ones(2, 3))
print(torch.arange(0, 3, step=0.5))

print("\n3. 타입 변경")
# torch.FloatTensor(행, 열)
# torch.FloatTensor(리스트)
# tensor.type_as(타입)
print(torch.FloatTensor(2, 3))
print(torch.FloatTensor([2, 3]))
print(torch.FloatTensor([2, 3]).type_as(torch.IntTensor()))

print("\n4. Numpy 호환")
# torch.from_numpy(numpy 행렬)
# tensor.numpy()
x1 = np.ndarray(shape=(2, 3), dtype=int, buffer=np.array([1, 2, 3, 4, 5, 6]))
print(x1)
x2 = torch.from_numpy(x1)
print(x2)
x3 = x2.numpy()
print(x3)

print("\n5. 텐서 위치 CPU에서, GPU에서")
# tensor.cuda()
x_cpu = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(x_cpu)
x_gpu = x_cpu.cuda()
print(x_gpu)

print("\n6. 텐서 크기")
# tensor.size()
print(torch.FloatTensor(10, 12, 3, 3).size())
print(torch.FloatTensor(10, 12, 3, 3).size()[:])
print(torch.FloatTensor(10, 12, 3, 3).size()[1:2])

print("\n7. 인덱싱")
# tensor[원하는 인덱스]
# torch.index_select(원본, 차원, 원하는 인덱스)
# torch.masked_select(원본, 마스크 행렬)
x = torch.rand(4, 3)
print(x)
print(x[[0, 3]])
print(torch.index_select(x, 0, torch.LongTensor([0, 3])))
mask = torch.ByteTensor(
    [[0, 0, 1],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]]
)
print(torch.masked_select(x, mask))

print("\n8. 병합")
# torch.cat(텐서 리스트, 병합 대상 차원)
# torch.stack(텐서 리스트, 쌓는 대상 차원)
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])  # 2행 3열 (x행 y열)
y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])  # 2행 3열 (i행 j열)
print(torch.cat([x, y], dim=0))  # 차원 0 (행) 에서 이어 붙임, 4행 3열 (x+i행 원본열)
print(torch.cat([x, y], dim=1))  # 차원 1 (열) 에서 이어 붙임, 2행 6열 (원본행 y+j열)
print(torch.stack([x, x, x, x, x], dim=0))  # 차원 0 (새로운 차원)에 쌓기, '5층' 2행 3열
print(torch.stack([x, x, x, x, x], dim=1))  # 차원 1 (행)에 쌓기, 2층 '5행' 3열
print(torch.stack([x, x, x, x, x], dim=2))  # 차원 2 (열)에 쌓기, 2층 3행 '5열'

print("\n9. 자르기")
# torch.chunk(텐서, 조각 갯수, 자르는 차원)
# torch.split(텐서, 조각 크기, 자르는 차원)
x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6],
                       [-1, -2, -3],
                       [-4, -5, -6]])
x_1, x_2, x_3 = torch.chunk(x, 3, dim=1)
print(x_1)
print(x_2)
print(x_3)
x_1, x_2 = torch.chunk(x, 2, dim=1)
print(x_1)
print(x_2)

print("\n10. 크기 1인 차원 없애거나 다시 만들기")
# torch.squeeze(원본, 검사 차원=None)
# torch.unsqueeze(원본, 살릴 차원=None)
print(torch.FloatTensor(10, 1, 3, 1, 4).size())
print(torch.FloatTensor(10, 1, 3, 1, 4).squeeze().size())
print(torch.FloatTensor(10, 1, 3, 1, 4).squeeze().unsqueeze(dim=0).size())

print("\n11. 초기화")
print(init.uniform_(torch.FloatTensor(3, 4), a=0, b=9))
print(init.normal_(torch.FloatTensor(3, 4), std=0.2))
print(init.constant_(torch.FloatTensor(3, 4), 7.777777))

print("\n12. 사칙연산")
x1 = torch.FloatTensor([[1, 2], [3, 4]])
x2 = torch.FloatTensor([[1, 2], [3, 4]])
print(torch.add(x1, x2))
print(x1+x2)  # 행렬 요소별 덧셈
print(x1-x2)  # 행렬 요소별 뺄셈
print(x1*x2)  # 행렬 요소별 곱셈
print(x1/x2)  # 행렬 요소별 나눗셈
print(x1+10)  # 자동 브로드캐스팅
print(x1-10)  # 자동 브로드캐스팅
print(x1*5)  # 자동 브로드캐스팅
print(x1/5)  # 자동 브로드캐스팅

print("\n13. 함수연산")
# torch.exp(텐서)
# torch.log(텐서)
x1 = torch.FloatTensor([[1, 2], [3, 4]])
print(x1**2)  # 행렬 요소별 거듭제곱
print(torch.exp(x1))  # 지수함수
print(torch.log(x1))  # 자연로그함수

print("\n14. 행렬곱")
# torch.mm(텐서, 텐서) -> 행렬곱
# torch.bmm(텐서, 텐서) -> 행렬곱 여러개
# torch.dot(텐서, 텐서) -> 행렬 내적
x1 = torch.FloatTensor([[1, 2], [3, 4]])
x2 = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.mm(x1, x2))  # 행렬 곱셈
x1 = torch.FloatTensor(10, 3, 4)  # 3행 4열 행렬 10층
x2 = torch.FloatTensor(10, 4, 5)  # 3행 4열 행렬 10층
print(torch.bmm(x1, x2).size())  # 여러 개의 행렬 곱셈 동시에
x1 = torch.FloatTensor([123, 456])
x2 = torch.FloatTensor([987, 654])
print(torch.dot(x1, x2))  # 행렬 내적

print("\n15. 행렬연산")
# tensor.t() -> 전치행렬
# tensor.transpose(전치할 인덱스)
# tensor.eig(고유벡터=False) -> 고윳값
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.t())  # 전치행렬
x = torch.FloatTensor(10, 3, 4)
print(x.transpose(1, 2).size())  # 특정 인덱스 전치
x = torch.FloatTensor(4, 4)
print(x.eig(eigenvectors=True))
