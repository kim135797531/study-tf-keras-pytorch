# -*- coding: utf-8 -*-

# noinspection PyMethodParameters
import torch

from utils_kdm.singleton import Singleton


# noinspection PyMethodParameters
class ManageDevice(Singleton):

    def __init__(cls):
        # TODO: 접근은 모든 클래스에서 가능 / 설정은 TrainerMetadata 에서
        # TODO: __init__ <-> TrainerMetadata 순환 참조 임시 해결
        super().__init__()
        cls.device = None

    def get(cls, call_from=''):
        # if call_from is not 'TrainerMetadata':
        #     print("TODO: 중요: 설정하는 주체가 TrainerMetadata 인지 확인하는 코드 넣기")
        return cls.device

    def set(cls, force_cpu=False, call_from=''):
        if call_from is not 'TrainerMetadata':
            print("TODO: 중요: 설정하는 주체가 TrainerMetadata 인지 확인하는 코드 넣기")
        if force_cpu or not torch.cuda.is_available():
            cls.device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            cls.device = torch.device('cuda:0')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
