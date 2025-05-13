from typing import Dict, Iterator

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import spaces

from agent.model.base import BaseModel

import re


class RolloutStorage:
    def __init__(
            self,
            nstep: int,
            nproc: int,
            observation_space: spaces.Box,
            action_space: spaces.Discrete,
            hidsize: int,
            device: th.device,
            tg_max_seq_len: int,
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        # Get obs shape and action dim
        assert isinstance(observation_space, spaces.Box)
        # assert isinstance(action_space, spaces.Discrete)
        obs_shape = getattr(observation_space, "shape")
        action_shape = (1,)

        # Tensors
        self.obs = th.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.actions = th.zeros(nstep, nproc, *action_shape, device=device).long()
        self.rewards = th.zeros(nstep, nproc, 1, device=device)
        self.masks = th.ones(nstep + 1, nproc, 1, device=device)
        self.vpreds = th.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = th.zeros(nstep, nproc, 1, device=device)
        self.returns = th.zeros(nstep, nproc, 1, device=device)
        self.advs = th.zeros(nstep, nproc, 1, device=device)
        self.successes = th.zeros(nstep + 1, nproc, 22, device=device).long()
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = th.zeros(nstep + 1, nproc, hidsize, device=device)
        self.text_obs_emd = th.zeros(nstep + 1, nproc, tg_max_seq_len, device=device)
        self.goals_emd = th.zeros(nstep + 1, nproc, tg_max_seq_len, device=device)

        self.transition_text_des = [[] for i in range(nstep + 1)]
        self.text_obs_des = [[] for i in range(nstep + 1)]
        self.action_des = [[] for i in range(nstep + 1)]
        self.goal_str = [[] for i in range(nstep + 1)]
        # Step
        self.step = 0

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_interval_steps_data(self, start: int, end: int):
        return self.transition_text_des[start:end]

    def get_inputs(self, step: int):
        inputs = {"obs": self.obs[step], "states": self.states[step], "text_obs_emd": self.text_obs_emd[step],
                  "goals_emd": self.goals_emd[step], 'text_obs_des': self.text_obs_des[step], 'goal_str':self.goal_str[step]}
        # inputs = {"obs": self.obs[step], "states": self.states[step]}  # ppoad
        return inputs

    def insert_transition_text_obs(self, step, transition_text_obs):
        self.transition_text_des[step] = transition_text_obs

    def insert(
            self,
            obs: th.Tensor,
            latents: th.Tensor,
            actions: th.Tensor,
            rewards: th.Tensor,
            masks: th.Tensor,
            vpreds: th.Tensor,
            log_probs: th.Tensor,
            successes: th.Tensor,
            text_obs_emd,
            goals_emd,
            text_obs_des,
            model: BaseModel,
            transition_text_desc=None,
            **kwargs,
    ):
        # Get prev successes, timesteps, and states
        prev_successes = self.successes[self.step]
        prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]

        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if new achievment is unlocked
        success_conds = successes != prev_successes
        success_conds = success_conds.any(dim=-1, keepdim=True)
        if success_conds.any():
            with th.no_grad():
                # 判断text_obs_emd是不是np.ndarray，如果是，则转成tensor
                if isinstance(text_obs_emd, np.ndarray):
                    text_obs_emd = th.tensor(text_obs_emd, device=self.device)
                if isinstance(goals_emd, np.ndarray):
                    goals_emd = th.tensor(goals_emd, device=self.device)
                # 将text_obs_emd和goals_emd放到与obs相同的设备上
                text_obs_emd = text_obs_emd.to(obs.device)
                goals_emd = goals_emd.to(obs.device)
                next_latents = model.encode({"obs": obs, "text_obs_emd": text_obs_emd,   # PPO
                                             "goals_emd": goals_emd})
                # next_latents = model.encode(th.tensor(obs))   # PPOAD
            states = next_latents - latents
            states = F.normalize(states, dim=-1)
            states = th.where(success_conds, states, prev_states)
        else:
            states = prev_states

        # Update successes, timesteps, and states if done
        done_conds = masks == 0    # ppo
        # done_conds = th.tensor(masks) == 0  # ppoad
        successes = th.where(done_conds, 0, successes)
        timesteps = th.where(done_conds, 0, timesteps)
        states = th.where(done_conds, 0, states)

        # Update tensors
        self.obs[self.step + 1].copy_(obs)    # ppo
        # self.obs[self.step + 1].copy_(th.tensor(obs))   # ppoad
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)
        self.text_obs_des[self.step + 1] = text_obs_des

        if isinstance(text_obs_emd, np.ndarray):
            text_obs_emd = th.tensor(text_obs_emd, device=self.device)
        if isinstance(goals_emd, np.ndarray):
            goals_emd = th.tensor(goals_emd, device=self.device)
        # 将text_obs_emd和goals_emd放到与obs相同的设备上
        text_obs_emd = text_obs_emd.to(obs.device)
        goals_emd = goals_emd.to(obs.device)
        self.text_obs_emd[self.step + 1].copy_(text_obs_emd.clone().detach())
        self.goals_emd[self.step + 1].copy_(goals_emd.clone().detach())

        # Update step
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        # Reset tensors
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])

        # Reset step
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        # Compute returns
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                    self.rewards[step]
                    + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                    - self.vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae

        # Compute advantages
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # Sample batch
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
        states = self.states[:-1].view(-1, *self.states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])
        text_obs_emd = self.text_obs_emd[:-1].view(-1, *self.text_obs_emd.shape[2:])
        goals_emd = self.goals_emd[:-1].view(-1, *self.goals_emd.shape[2:])

        for indices in sampler:
            batch = {
                "obs": obs[indices],
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices],
                "log_probs": log_probs[indices],
                "advs": advs[indices],
                "text_obs_emd": text_obs_emd[indices],
                "goals_emd": goals_emd[indices],
            }
            yield batch

    def input_info_extract(self, Input):
        pattern = re.compile(r"Player sees: <(.+?)>\nPlayer status: <(.+?)>\nPlayer inventory: <(.+?)>")
        match = pattern.match(Input)

        if match:
            observation = match.group(1)
            status = match.group(2)
            inventory = match.group(3)

            return observation, status, inventory

        else:
            return None

