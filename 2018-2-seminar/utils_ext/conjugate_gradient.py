# -*- coding: utf-8 -*-

# RLKR에서 차용한 걸 다시 차용
# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py

import torch
from utils_kdm.trainer_metadata import TrainerMetadata


def conjugate_gradient(func_Ax, s_batch, loss_grad_data, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312

    공역 구배법 = 켤레 기울기법
    == 설명 (from 위키피디아 한국어판) ==
    Ax = b 를 풀고 싶다고 하자. 이 때 A는 (대칭행렬)이며 (대칭행렬의 이차형식 > 0)이어야 한다.
    켤레 벡터 뜻은 대충 A의 이차형식을 0으로 만드는 두 벡터임..

    아무튼 정답벡터 x를 (x = α*켤레벡터들의 합)으로 나타낼 수 있다고 하면
    어찌어찌 잘 하면 α = <p, b> / |p|^2 으로 나타낼 수 있다고 함

    근데 차원이 커지면 이거 풀기 귀찮으니까 대충 때려 맞추고 점점 해답에 접근하는 방법을 씀
    즉 r0 = b - Ax0 라 하고 이걸로 α 구함
    이 α를 기준으로 x와 r을 다시 나타냄
    다시 나타낸 r로 잘 구함... (반복)

    대충 이런 개념

    z = Ax
    v =  r*r
        -----
        p*(Ax)

    new_x = old_x + v * p
    new_r = old_r + v * Ax
    meow = new_r * new_r
           -------------
               r * r
    new_p = r + (meow * p)
    """
    device = TrainerMetadata().device

    p = loss_grad_data.clone().to(device)
    r = loss_grad_data.clone().to(device)
    x = torch.zeros_like(loss_grad_data).to(device)
    dot_rr = torch.dot(r, r).to(device)

    for i in range(cg_iters):
        # 원래 여기서 z = Ax를 계산해야 한다
        # 그런데 A를 갖고 있기 힘드니깐 대충 Ax 예상해서 던져주는 놈을 사용할 것이다
        # z = A * p
        # z = func_Ap
        z = func_Ax((p, s_batch))
        alpha = dot_rr / torch.dot(p, z).to(device)
        x += alpha * p
        r -= alpha * z
        new_dot_rr = torch.dot(r, r).to(device)
        meow = new_dot_rr / dot_rr
        p = r + (meow * p)

        dot_rr = new_dot_rr
        if dot_rr < residual_tol:
            break

    return x
