from typing import Dict

import torch as th

from gym import spaces

from agent.model.base import BaseModel
from agent.impala_cnn import ImpalaCNN
from agent.action_head import CategoricalActionHead
from agent.mse_head import ScaledMSEHead
from agent.torch_util import FanInInitReLULayer
import torch.nn as nn
from encoder import SbertEncoder
import torch



class PPOModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        device="cuda",
        sbert_path=None,
    ):
        super().__init__(observation_space, action_space, device)
        self.device = device
        # Encoder
        obs_shape = getattr(self.observation_space, "shape")
        # self.enc = Encoder(obs_shape, hidsize, device, dense_init_norm_kwargs, impala_kwargs, other_dim=0)
        self.image_enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        self.lang_goal_encoder = SbertEncoder(hidsize, device, path=sbert_path).to(self.device)
        outsize = impala_kwargs["outsize"]
        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.hidsize = hidsize * 2

        # Heads
        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=hidsize * 2,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=hidsize * 2,
            outsize=1,
            **mse_head_kwargs,
        )

    @th.no_grad()
    def act(self,input) -> Dict[str, th.Tensor]:   # ppo
        # Check training mode
        assert not self.training
        # Pass through model
        outputs = self.forward(input)

        # Sample actions
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)

        # Compute log probs
        log_probs = self.pi_head.log_prob(pi_logits, actions)

        # Denormalize vpreds
        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)

        # Update outputs
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})

        return outputs

    # def act(self,obs, states) -> Dict[str, th.Tensor]:         # ppoad
    #     # Check training mode
    #     assert not self.training
    #     # Pass through model
    #     # outputs = self.forward(input['obs'], input['states'])
    #     outputs = self.forward(obs, states)
    #
    #     # Sample actions
    #     pi_logits = outputs["pi_logits"]
    #     actions = self.pi_head.sample(pi_logits)
    #
    #     # Compute log probs
    #     log_probs = self.pi_head.log_prob(pi_logits, actions)
    #
    #     # Denormalize vpreds
    #     vpreds = outputs["vpreds"]
    #     vpreds = self.vf_head.denormalize(vpreds)
    #
    #     # Update outputs
    #     outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
    #
    #     return outputs

    def forward(self, input) -> Dict[str, th.Tensor]:
        # Pass through encoder
        latents = self.encode(input)

        # Pass through heads
        pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

        return outputs

    def encode(self, input) -> th.Tensor:
        # Pass through encoder
        # x = self.enc(input)
        x = self.image_enc(input['obs'])
        state_and_goal = torch.cat([input['text_obs_emd'], input['goals_emd']], dim=-1)
        sg_output = self.lang_goal_encoder(state_and_goal)
        x = self.linear(x)

        return torch.cat([x, sg_output], dim=-1)
        # PPOAD
        # x = self.image_enc(input)
        # x = self.linear(x)
        # return x

    def compute_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        text_obs_emd: th.Tensor,
        goals_emd: th.Tensor,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # Pass through model
        # outputs = self.forward(obs, **kwargs)    #PPOAD
        outputs = self.forward({'obs':obs, 'text_obs_emd': text_obs_emd, 'goals_emd': goals_emd})   # PPO

        # Compute policy loss
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()

        # Compute entropy
        entropy = self.pi_head.entropy(pi_logits).mean()

        # Compute value loss
        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()

        # Define losses
        losses = {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

        return losses
