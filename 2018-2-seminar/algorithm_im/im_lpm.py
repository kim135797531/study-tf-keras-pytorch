# -*- coding: utf-8 -*-

import utils_kdm as u
from algorithm_im.im_base import IntrinsicMotivation
from algorithm_im.region import RegionManager


class LearningProgressMotivationOudeyer(IntrinsicMotivation):
    # Oudeyer et al. (2007)

    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self._set_hyper_parameters()
        self.region_manager = RegionManager(self.state_size, self.action_size)

        self.register_serializable([
            'self.region_manager',
        ])

    def _set_hyper_parameters(self):
        super()._set_hyper_parameters()

    def intrinsic_motivation_impl(self, i_episode, step, current_sars, current_done):
        # Learning progress motivation (LPM)
        current_state, current_action, current_reward, current_next_state = current_sars

        examplar = self.region_manager.exemplar_structure(
            u.t_float32(current_state),
            u.t_float32(current_action),
            u.t_float32(current_next_state)
        )
        self.region_manager.add(examplar)

        region = self.region_manager.find_region(examplar)
        past_error = region.get_past_error_mean()
        current_error = region.get_current_error_mean()
        intrinsic_reward = past_error - current_error

        # TODO: 환경 평소 보상 (1) 정도로 clip 해줄까?
        # intrinsic_reward_batch = torch.clamp(intrinsic_reward_batch, min=-2, max=2)
        # self.viz.draw_line(y=torch.mean(intrinsic_reward_batch), interval=1000, name="intrinsic_reward_batch")

        return intrinsic_reward
