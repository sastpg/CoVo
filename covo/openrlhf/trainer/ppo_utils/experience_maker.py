import time
import json
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

from .score import compute_RiV, compute_RiL, kl_divergence, process_thoughts, detect_duplicate
from .eval import check, last_boxed_only_string
import numpy as np
import math

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]
    rewards: list[str]
    curiosity_rewards: list[str]
    real_rewards: list[str]


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        
        if args.enable_accuracy_filter:
            experiences = self.filter(experiences)

        if not experiences:
            return []

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
            else:
                r = remote_rm_fn(
                    self.remote_rm_url, queries=queries, prompts=samples.prompts, labels=samples.labels
                ).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
    
    @torch.no_grad()
    def filter(self, experiences: List[Experience]) -> List[Experience]:
        """
        Filter experiences based on accuracy reward.

        Output:
        - experiences: List of filtered Experience
        """

        args = self.strategy.args
        accuracy_rewards = torch.cat([experience.info["reward"] for experience in experiences])
        accuracy_rewards = accuracy_rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        accuracy_rewards = torch.mean(accuracy_rewards, dim=-1)

        assert len(experiences) % len(accuracy_rewards) == 0
        group_len = len(experiences) // len(accuracy_rewards)
        grouped_experience = [experiences[i : i + group_len] for i in range(0, len(experiences), group_len)]
        filtered_experiences = []
        for group, reward in zip(grouped_experience, accuracy_rewards):
            if 0.001 <= reward.item():
                filtered_experiences.extend(group)

        print(f"Filtered {len(experiences) - len(filtered_experiences)} experiences.")

        return filtered_experiences

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_labels, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)


        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(
            sequences,
            num_actions,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start

        base_action_log_probs, value = ref_values[0], ref_values[1]
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        assert self.strategy.ring_attn_group is None
        # broadcast rewards to all ring attention ranks when using remote RM
        if self.remote_rm_url and self.strategy.ring_attn_group is not None:
            if self.strategy.ring_attn_rank == 0:
                dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                dist.broadcast_object_list(
                    rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )

        # rewards
        all_rewards = torch.tensor(samples.rewards).to(device)
        r = all_rewards
        curi_r = torch.tensor(samples.curiosity_rewards).to(device)
        real_r = torch.tensor(samples.real_rewards).to(device)
        other_rewards = {"curiosity_rewards": curi_r, "real_rewards": real_r}
        print(f"int_r: {r}, curi_r: {curi_r}, real_r: {real_r}")

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            **other_rewards
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience
    
    def _generate_resp(self, all_prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=8192,
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        resp = []
        for output in all_outputs:
            resp.append(output.outputs[0].text)
        return resp

    def _get_rewards(self, all_prompts, all_outputs, all_labels):
        """
        Calculate rewards for outputs.

        This function processes the outputs generated from multiple prompts, compares them with labels,
        and calculates rewards based on the consistency and volatility of the reasoning trajectory.

        Parameters:
        - all_prompts: A list of all prompts used to generate outputs.
        - all_outputs: A list of outputs generated from the prompts.
        - all_labels: A list of ground-truth labels corresponding to each prompt.

        Returns:
        - all_intrinsic_rewards: A list of intrinsic rewards calculated for each output.
        - all_curiosities: A list of curiosity rewards for each output, reflecting the diversity of thoughts.
        - all_real_rewards: A list of ground-truth rewards indicating whether the output matches the true label.
        """
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]
        
        args = self.strategy.args
        all_intrinsic_rewards = [] # rollout_num
        all_real_rewards = []  # rollout_num x n_samples_per_prompt
        all_answer_index = []  # rollout_num x n_samples_per_prompt
        all_answer_len = []  # rollout_num x n_samples_per_prompt
        all_thoughts_step_num = []  # rollout_num x n_samples_per_prompt
        all_independ_answer_num = [] # rollout_num
        all_states_with_question = []
        all_states_with_question_ori_len = []
        # Process each output
        for j in range(0, len(all_outputs), args.n_samples_per_prompt):
            outputs = all_outputs[j : j + args.n_samples_per_prompt]
            model_input = all_prompts[j]
            ground_truth = all_labels[j]

            answer_index = []
            answer_len = []
            independ_answer_list = []
            thoughts = []
            thoughts_step_num = []
            real_rewards = []
            # 提取答案
            for output in outputs:
                output_text = output.outputs[0].text
                pred = last_boxed_only_string(output_text)
                if not pred:
                    answer_index.append(None)
                    answer_len.append(999)
                    real_rewards.append(False)
                else:
                    answer_idx = 0
                    for ans in independ_answer_list:
                        if check(ans, pred):
                            break
                        answer_idx += 1
                    assert answer_idx <= len(independ_answer_list)
                    if answer_idx == len(independ_answer_list):
                        independ_answer_list.append(pred)
                    
                    answer_index.append(answer_idx)
                    answer_len.append(len(independ_answer_list[answer_idx]))
                    real_rewards.append(check(ground_truth, pred))
                thoughts.append(output_text)

            all_answer_index.append(answer_index)
            all_answer_len.append(answer_len)
            all_real_rewards.append(real_rewards)
            all_independ_answer_num.append(len(independ_answer_list) + 1)  # 第一列是curiosity
            # breakpoint()
            # S_i -> S_{i+1}, y_1, y_2, ..., y_K,
            # d(s_i, s_{i+1}), d(s_i, y_1), d(s_i, y_2), ..., d(s_i, y_k)
            for thought in thoughts:
                steps = process_thoughts(thought)
                thoughts_step_num.append(len(steps))
                for k in range(len(steps)):
                    all_states_with_question.append(model_input + "\n".join(steps[:k+1]))
                    all_states_with_question_ori_len.append([len(model_input + "\n".join(steps[:k])), 1])   # 这里列表每个item最后的 1/0  用来表示计算curiosity reward 还是intrinsic reward
                    for y in independ_answer_list:
                        all_states_with_question.append(model_input + "\n".join(steps[:k+1]) + "\nThe answer is: " + y)
                        all_states_with_question_ori_len.append([len(model_input + "\n".join(steps[:k+1])), 0])
            
            # breakpoint()
            all_thoughts_step_num.append(thoughts_step_num)

        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            skip_special_tokens=False,
            prompt_logprobs=1,
            logprobs=1,
            include_stop_str_in_output=True,
        )

        all_states_with_question_token_ids = self.tokenize_fn(all_states_with_question, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_states_with_question_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            states_prompt_token_ids = all_states_with_question_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=states_prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_output_for_logprobs = sum(ray.get(all_output_refs), [])
        

        del all_states_with_question
        all_matrix = []
        for output_logprob, ori_len in zip(all_output_for_logprobs, all_states_with_question_ori_len):
            prompt_logprobs = output_logprob.prompt_logprobs
            prompt_token_ids = output_logprob.prompt_token_ids

            cumulative_text = "<|im_start|>"  # Qwen, LLama not same
            tokens_logprobs = []
            for logprob_dict, token_id in zip(prompt_logprobs[1:], prompt_token_ids[1:]):
                cumulative_text += logprob_dict[token_id].decoded_token
                if len(cumulative_text) > ori_len[0]:
                    logprobs = logprob_dict[token_id].logprob
                    tokens_logprobs.append(logprobs)
            # debug
            # print(cumulative_text)
            # print(cumulative_text[ori_len[0]:])
            # breakpoint()
            # end debug
            tokens_logprobs.append(output_logprob.outputs[0].cumulative_logprob)
            avg_logprobs = np.mean(tokens_logprobs)
            perplexity = math.exp(-avg_logprobs)
            # 如果这一项代表curiosity rewards, 那么减去KL(P,U)
            if ori_len[1] == 1:
                kl_penalty = kl_divergence(tokens_logprobs)
                perplexity = perplexity - kl_penalty
            all_matrix.append(perplexity)
        all_matrix = np.array(all_matrix)


        # 循环内是同一个prompt采样的n组数据，计算intrinsic rewards
        all_consistencies, all_volatilities, all_curiosities, all_group_data = [], [], [], []
        states_idx = 0
        for thoughts_step_num, independ_answer_num, answer_index, answer_len, real_rewards in zip(all_thoughts_step_num, all_independ_answer_num, all_answer_index, all_answer_len, all_real_rewards):
            all_states_num = sum(thoughts_step_num) * independ_answer_num
            # This is the distance matrix for **only one trajectory**
            # The shape of distance matrix is (States Number x The number of distinct answers sampled under the same prompt)
            # 请注意这里的第一列是 s_i -> s_{i+1} 的 distance, 第二列是 s_i -> answer y_1的distance, 第三列是 s_i -> answer y_2的distance, ...
            # 矩阵的每一行是不同的states, 每一列是不同的答案
            distance_matrix = all_matrix[states_idx:states_idx+all_states_num].reshape(sum(thoughts_step_num), independ_answer_num)
            states_idx += all_states_num

            curiosity_matrix = distance_matrix[:, 0]
            fig_data = distance_matrix[:, 1:] # 只保留 s_i到final answer y 的distance，s_i -> s_{i+1}的distance是计算curiosity reward 的，计算 intrinsic reward 不需要
            fig_data = fig_data / np.linalg.norm(fig_data, axis=1, ord=1, keepdims=True) # normalize the D (S, Y)

            # the same prompt, loop for each answer
            group_data = {}
            min_answer_len = min(answer_len)
            idx = 0
            consistencies, volatilities, curiosities = np.zeros(args.n_samples_per_prompt), np.zeros(args.n_samples_per_prompt), np.zeros(args.n_samples_per_prompt)
            for target_idx, (num_thoughts, target_column, ans_len, is_correct) in enumerate(zip(thoughts_step_num, answer_index, answer_len, real_rewards)):
                curiosity_list = curiosity_matrix[idx:idx+num_thoughts]
                curiosity = np.mean(curiosity_list)
                
                embeddings = fig_data[idx:idx+num_thoughts, :]
                idx += num_thoughts
                ratios = []

                if target_column == None:
                    continue
                if group_data.get(target_column) is None:
                    group_data[target_column] = {"idx": [], "consistency": [], "volatility": [], "curiosity": [], "is_correct": is_correct, "answer_len": ans_len / min_answer_len}
                # only one answer
                if independ_answer_num <= 2:
                    group_data[target_column]["idx"].append(target_idx)
                    group_data[target_column]["consistency"].append(0)
                    group_data[target_column]["volatility"].append(0)
                    group_data[target_column]["curiosity"].append(curiosity)
                    continue


                # Compute consistency and volatility for intrinsic rewards, please refer to Alg.1 in the paper
                for st in embeddings:
                    target_value = st[target_column]
                    other_values = st[:target_column].tolist() + st[target_column + 1:].tolist()
                    min_other_value = min(other_values)
                    ratio = target_value / min_other_value
                    ratios.append(ratio)
                
                consistency = len([x for x in ratios if x < 1]) / len(ratios)
                converage_number = 0
                for value in reversed(ratios):
                    if value > 1:
                        break
                    converage_number += 1
                converage = converage_number / len(ratios)
                
                # update info
                group_data[target_column]["idx"].append(target_idx)
                group_data[target_column]["consistency"].append(consistency)
                group_data[target_column]["volatility"].append(1-converage)   # Note: volatility = 1-converage
                group_data[target_column]["curiosity"].append(curiosity)
                
            
            # group score
            intrinsic_rewards = np.zeros(args.n_samples_per_prompt)
            reward_idx = []
            for key, _ in group_data.items():
                group_data[key]["riv"] = compute_RiV(group_data[key]["consistency"], group_data[key]["volatility"])
                group_data[key]["ril"] = compute_RiL(group_data[key]["consistency"], group_data[key]["volatility"])
                # assign rewards
                # NOTE: You can also try: intrinsic_rewards[group_data[key]["idx"]] = group_data[key]["riv"]
                if args.intrinsic_reward == "riv":
                    bound = 0.5
                elif args.intrinsic_reward == "ril":
                    bound = 0.05
                else:
                    raise NotImplementedError
                if group_data[key][args.intrinsic_reward] >= bound:
                    group_data[key]["judge"] = True
                    reward_idx.extend(group_data[key]["idx"])
                else:
                    group_data[key]["judge"] = False
                consistencies[group_data[key]["idx"]] = group_data[key]["consistency"]
                volatilities[group_data[key]["idx"]] = group_data[key]["volatility"]
                curiosities[group_data[key]["idx"]] = group_data[key]["curiosity"]
            
            intrinsic_rewards[reward_idx] = 1

            all_intrinsic_rewards.extend(intrinsic_rewards.tolist())
            all_consistencies.extend(consistencies.tolist())
            all_volatilities.extend(volatilities.tolist())
            all_curiosities.extend(curiosities.tolist())
            all_group_data.append(group_data)

        assert states_idx == len(all_matrix)
        assert len(all_consistencies) == len(all_outputs)
        ## record sample information
        all_real_rewards = [item for sublist in all_real_rewards for item in sublist]
        for pro, out, gt, cons, vol, curi in zip(all_prompts, all_outputs, all_real_rewards, all_consistencies, all_volatilities, all_curiosities):
            with open(f"{args.save_output_path}/{args.intrinsic_reward}_outputs.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps({"prompt": pro, "output": out.outputs[0].text, "gt": gt, "consistency": cons, "volatility": vol, "curiosity": curi}) + "\n")
        
        with open(f"{args.save_output_path}/{args.intrinsic_reward}_group.jsonl", "a", encoding='utf-8') as f:
            for gd in all_group_data:
                f.write(json.dumps(gd) + "\n")
        
        return all_intrinsic_rewards, all_curiosities, all_real_rewards

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        # custom rewards
        all_intrinsic_rewards, all_curiosity_values, all_ground_truths = self._get_rewards(all_prompts, all_outputs, all_labels)
        assert len(all_intrinsic_rewards) == len(all_outputs)

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            intrinsic_rewards = all_intrinsic_rewards[i : i + self.strategy.args.micro_rollout_batch_size]
            curiosity_values = all_curiosity_values[i : i + self.strategy.args.micro_rollout_batch_size]
            groud_truths = all_ground_truths[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                rewards = []
                curiosity_rewards = []
                real_rewards = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                    # compute reward
                    int_r = intrinsic_rewards[i]
                    curi_r = 0.4 if args.enable_curiosity and curiosity_values[i] > 2 else 0.0
                    real_r = 1.0 if groud_truths[i] else 0.0
                    rewards.append(int_r + curi_r)
                    curiosity_rewards.append(curi_r)
                    real_rewards.append(real_r)


                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                        rewards=intrinsic_rewards,
                        curiosity_rewards=curiosity_rewards,
                        real_rewards=real_rewards,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
