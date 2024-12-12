import logging
import time
from abc import ABC
import os 
import re
from copy import deepcopy
import json
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union
from tqdm import trange
import ray
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from mcts.run_src.mcts_backbone import MCTS_Searcher
from mcts.run_src.rstar_utils import Node_Type, stochastic_find_best_solution, print_tree_from_root
from mcts.run_src.mcts_for_reasoning import Generator as MCTSGenerator, Reasoning_MCTS_Node
from mcts.run_src.mcts_only_ost_reasoning import OstOrAnswerGenerator, OstOrAnswerNode


logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.advantages = to(self.advantages, device)
        if self.returns is not None:
            self.returns = to(self.returns, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        
        self.advantages = pin_memory(self.advantages)
        if self.returns is not None:
            self.returns = pin_memory(self.returns)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
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
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    rewards: Optional[torch.Tensor] = None
    token_counters: Optional[list[int]] = None
    call_counters: Optional[list[int]] = None


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
        remote_rm_url: str = None,
        reward_fn=None,
        critic_tokenizer=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.critic_tokenizer = critic_tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if not padding:
            # when padding is False, return tokenized texts as list
            return tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_solutions: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        experiences = []
        for samples in tqdm(
            self.generate_samples(all_prompts, all_solutions, **generate_kwargs),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples))

        experiences = self.process_experiences(experiences)
        # calculate return and advantages
        if self.advantage_estimator == 'group_norm':
            experiences = self.get_grpo_advantages(experiences, args.n_samples_per_prompt)
        else:
            for experience in experiences:
                num_actions = experience.info["num_actions"]
                reward = compute_reward(
                    experience.info["reward"],
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
                elif self.advantage_estimator == "reinforce":
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
        for experience in experiences:
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
        return experiences
    
    @torch.no_grad()
    def get_grpo_advantages(self, experiences: List[Experience], n_samples_per_prompt: int) -> List[Experience]:
        use_prm = self.strategy.args.use_prm
        all_advantages = []
        if use_prm:
            micro_rollout_bsz = len(experiences[0].info["reward"])
            all_rewards = [reward for experience in experiences for reward in experience.info["reward"]]
        else:
            micro_rollout_bsz = experiences[0].info["reward"].size(0)
            # 16个(4,1) -> (64,1)
            all_rewards = torch.concat([experience.info["reward"] for experience in experiences])
        if use_prm:
            for i in range(0, len(all_rewards), n_samples_per_prompt):
                rewards = all_rewards[i: i + n_samples_per_prompt]
                returns = []
                for reward in rewards:
                    cumsum_reverse = torch.cumsum(reward.flip(0), dim=0).flip(0)
                    returns.append(cumsum_reverse)
                cat_returns = torch.cat(returns)
                mean = cat_returns.mean()
                std = cat_returns.std()
                for _return in returns:
                    all_advantages.append((_return-mean)/(std+1e-8))
            # 改成(-1, micro_batchsize)
            _all_advantages = []
            for i in range(0, len(all_advantages), micro_rollout_bsz):
                _all_advantages.append(all_advantages[i: i + micro_rollout_bsz])
            all_advantages = _all_advantages
        else:
            for i in range(0, all_rewards.size(0), n_samples_per_prompt):
                # (4,1)
                rewards = all_rewards[i: i + n_samples_per_prompt]
                mean = rewards.mean()
                std = rewards.std()
                advantages = (rewards - mean) / (std + 1e-8)
                all_advantages.append(advantages)
            # (64,1)
            all_advantages = torch.concat(all_advantages)
            all_advantages = all_advantages.reshape(-1, micro_rollout_bsz)
            # (16,4)
            all_advantages = torch.unbind(all_advantages)
        if not self.packing_samples:
            for i, experience in enumerate(experiences):
                experience.advantages = all_advantages[i].unsqueeze(1).expand(
                    experiences[i].action_mask.shape).detach()
        else:
            for experience, advantages in zip(experiences, all_advantages):
                if use_prm:
                    experience.advantages = advantages
                    experience.info["reward"] = torch.tensor([reward.mean() for reward in experience.info["reward"]]).reshape(-1, 1)
                else:
                    packed_advantages = []
                    # 判断如果维度足够了，就不需要expand
                    for i, num_action in enumerate(experience.info["num_actions"]):
                        # num_action 每个不一样
                        packed_advantages.append(advantages[i].expand(num_action))
                    experience.advantages = packed_advantages
        return experiences
    
    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_solutions: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
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
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
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
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
        if self.advantage_estimator == 'group_norm':
            kl = None
        else:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1) if kl is not None else None,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "token_counter": sum(samples.token_counters)/len(samples.token_counters),
            "call_counter": sum(samples.call_counters)/len(samples.call_counters),
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> List[Experience]:
        # TODO: add more methods to process experiences
        return experiences

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
    def __init__(self, *args, vllm_generator, evaluator, vllm_engines: List = None, packing_samples=False, use_prm=False, mcts_use_prm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        # 用于非mcts情况下，用orm还是prm
        self.use_prm = use_prm
        # 用于mcts情况下，如果有prm可以使节点从中间返回
        self.mcts_use_prm = mcts_use_prm
        model, tokenizer = None, None
        args = self.strategy.args
        self.vllm_generator = vllm_generator
        if args.mcts_mode:
            if args.mcts_mode == 'only_ost':
                # TODO: shi
                self.generator = OstOrAnswerGenerator(
                    args, 
                    tokenizer, 
                    model, 
                    evaluator, 
                    vllm_generator, 
                    self.reward_model[0], 
                    mcts_use_prm=self.mcts_use_prm,
                    enable_go_explore=args.enable_go_explore,
                    multi_step_one_node=args.multi_step_one_node,
                    correct_step_thres=args.correct_step_thres
                )
            elif args.mcts_mode == 'full_actions':
                self.generator = MCTSGenerator(args, tokenizer, model, evaluator, vllm_generator)
        else:
            self.generator = None
        self.evaluator = evaluator


    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_solutions: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_solutions, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_solutions: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)
        if self.strategy.args.mcts_mode:
            return self._mcts_vllm(all_prompts, all_solutions, **generate_kwargs)
        else:
            return self._generate_vllm(all_prompts, **generate_kwargs)
    
    @torch.no_grad()
    def make_experience_prm(self, samples: Samples) -> Experience:
        km_token = 'ки'
        sep_token = '\n'
        km_token_id1 = 12902
        km_token_id2 = 1107
        good_token_id = 648
        bad_token_id = 387
        candidate_tokens = [good_token_id, bad_token_id]

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

        # rewards
        # 根据packed_seq_lens 分割成多段，进行decode
        sequences_split = []
        offset = 0
        tokens_list = sequences_cpu.tolist()[0]
        for length in packed_seq_lens:
            # 去掉eottoken
            sequences_split.append(tokens_list[offset : offset + length-1])
            offset += length
        decoded_sequences = self.tokenizer.batch_decode(sequences_split, skip_special_tokens=False)
        # ------------------------------prm 处理------------------------------
        # split_actor_p_responses = [resp.split(sep_token) for resp in decoded_sequences]
        # # 给每个response加上sep_token，过滤掉空字符串
        # # 去掉原来里面存在的km token
        # split_actor_p_responses = [[resp.replace(km_token, '')+sep_token for resp in split_resp if resp.strip()] for split_resp in split_actor_p_responses]
        # for split_resp in split_actor_p_responses:
        #     # 去掉最后一个换行符\n
        #     split_resp[-1] = split_resp[-1][:-1]
        
        # # TODO: 检查下有没有填充
        # split_actor_seqs = [[self.tokenize_fn(resp, self.prompt_max_len, device='cuda') for resp in split_resp] for split_resp in split_actor_p_responses]
        # # 元素为torch.tensor, (1, len)
        # split_actor_seqs = [[item['input_ids'][0] for item in split_seq] for split_seq in split_actor_seqs ]
        # # 用于确定每个step的reward位置
        # split_actor_lens = [[len(seq) for seq in split_seq] for split_seq in split_actor_seqs]
        # concatenated_actor_seqs = []
        # for split_actor_seq in split_actor_seqs:
        #     # 展平二维列表
        #     flat_list = [item for sublist in split_actor_seq for item in sublist]
        #     concatenated_actor_seqs.append(flat_list)
        # packed_seq_lens = [len(seq) for seq in concatenated_actor_seqs]
        # sequences = []
        # attention_mask = []
        # for i, seq in enumerate(concatenated_actor_seqs):
        #     sequences.extend(seq)
        #     attention_mask.extend([1]*len(seq))
        
        # sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
        # attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
        # sequences_cpu = sequences.to("cpu")
        # attention_mask_cpu = attention_mask.to("cpu")

        # # fix bug, 至少有一个，不然backward会没有更新报错
        # km_join_p_responses = [km_token.join(split_resp)+km_token for split_resp in split_actor_p_responses]
        # reward_encoded_inputs = self.tokenize_fn(km_join_p_responses, self.prompt_max_len*10, device='cuda', tokenizer=self.critic_tokenizer, padding=False)
        # step_rewards_ref = self.reward_model[0].get_prm.remote(reward_encoded_inputs)
        # step_rewards = ray.get(step_rewards_ref)
        # # step_rewards = self.get_step_rewards_ray(km_join_p_responses)
        # # 转成二维
        # step_rewards_2d = []
        # rewards = torch.zeros_like(sequences, device='cuda', dtype=torch.bfloat16)
        # start = 0
        # step_i = 0
        # for one_split_lens in split_actor_lens:
        #     step_reward = []
        #     for split_len in one_split_lens:
        #         idx = start + split_len - 1
        #         step_reward.append(step_rewards[step_i])
        #         rewards[0, idx] = step_rewards[step_i]
        #         start += split_len
        #         step_i += 1
        #     step_rewards_2d.append(step_reward)
        # # 如果len(reward)!=0，否则avg=0
        # avg_step_rewards = []
        # for reward in step_rewards_2d:
        #     if len(reward):
        #         avg_step_rewards.append(sum(reward)/len(reward))
        #     else:
        #         avg_step_rewards.append(0)
        rm_res_ref = self.reward_model[0].get_prm_res.remote(decoded_sequences, return_seqs_and_masks=True)
        rm_res = ray.get(rm_res_ref)
        packed_seq_lens = rm_res['packed_seq_lens']
        sequences = rm_res['sequences']
        attention_mask = rm_res['attention_mask']
        sequences_cpu, attention_mask_cpu = sequences.to("cpu"), attention_mask.to("cpu")
        rewards = rm_res['rewards']

        # ------------------------------prm 处理------------------------------
        seq_nums = len(packed_seq_lens)
        input_len = int((samples.total_length[0] - samples.response_length[0]).tolist())
        for i in range(seq_nums):
            output_len = packed_seq_lens[i]-input_len
            num_actions[i] = output_len
            samples.response_length[i] = output_len
            samples.total_length[i] = input_len + output_len

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)
        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start
        base_action_log_probs, value = ref_values[0], ref_values[1]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()
        
        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1) if kl is not None else None
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            rewards = unpacking_samples(rewards, packed_seq_lens)
            # num_actions 还得更新
            for i, num_action in enumerate(num_actions):
                rewards[i] = rewards[i][-num_action:]

            # (bacch_size, max_seq_len) 在末尾补0
            # rewards = pad_sequence(rewards, batch_first=True, padding_value=0.0)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions) if kl is not None else None
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device) if kl is not None else None
        info = {
            # "kl": kl_mean,
            "reward": rewards,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "token_counter": torch.tensor(samples.token_counters, dtype=torch.float32),
            "call_counter": torch.tensor(samples.call_counters, dtype=torch.float32),
        }
        if kl is not None:
            info['kl'] = kl_mean

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time
        experience = Experience(
            sequences,
            action_log_probs,
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
    
    def make_experience_mcts(self, samples: Samples) -> Experience:
        km_token = 'ки'
        km_token_id1 = 12902
        km_token_id2 = 1107
        good_token_id = 648
        bad_token_id = 387
        candidate_tokens = [good_token_id, bad_token_id]

        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        rewards = samples.rewards

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start
        base_action_log_probs, value = ref_values[0], ref_values[1]
        

        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        # (batch_size, 1)
        r = rewards

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()
        
        # if self.advantage_estimator == 'group_norm':
        #     kl = None
        # else:
        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1) if kl is not None else None
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions) if kl is not None else None
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device) if kl is not None else None
        info = {
            # "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "token_counter": torch.tensor(samples.token_counters, dtype=torch.float32),
            "call_counter": torch.tensor(samples.call_counters, dtype=torch.float32),
        }
        if kl is not None:
            info['kl'] = kl_mean

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time
        experience = Experience(
            sequences,
            action_log_probs,
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

    @torch.no_grad()
    def get_reward_prm_as_orm(self, qa_list:list[str])->List[float]:
        km_token = 'ки'
        good_token_id = 648
        bad_token_id = 387
        candidate_tokens = [good_token_id, bad_token_id]

        qa_list = [qa.replace(km_token, ' ')+km_token for qa in qa_list]
        critic_inputs = self.tokenize_fn(qa_list, self.prompt_max_len*10, device='cuda', tokenizer=self.critic_tokenizer, padding=False)
        critic_encoded_sequences = critic_inputs['input_ids']
        critic_attention_mask = critic_inputs['attention_mask']
        critic_packed_seq_lens = []
        # sequences 是二维的，获取每行长度，作为critic_packed_seq_lens
        for seq in critic_encoded_sequences:
            critic_packed_seq_lens.append(len(seq))
        
        # 展平critic_encoded_sequences, critic_encoded_mask为一维
        _critic_encoded_sequences = []
        for seq in critic_encoded_sequences:
            _critic_encoded_sequences.extend(seq)
        critic_encoded_sequences = torch.tensor(_critic_encoded_sequences).unsqueeze(0)
        _critic_attention_mask = []
        for i, mask in enumerate(critic_attention_mask):
            # _critic_attention_mask.extend(mask*(i+1))
            _critic_attention_mask.extend([v*(i+1) for v in mask])
        critic_attention_mask = torch.tensor(_critic_attention_mask).unsqueeze(0)
        r_ref = self.reward_model[0].forward.remote(critic_encoded_sequences, critic_attention_mask, packed_seq_lens=critic_packed_seq_lens)
        rewards = ray.get(r_ref)
        def process_reward(reward):
            # reward: (batch_size, seq_len, vocab_size)
            reward = reward[..., candidate_tokens]
            reward = reward.softmax(dim=-1)[...,0]
            reward = reward[:, -1]
            # step_rewards = [score[(reward_encoded_sequences[i]==km_token_id1) | (reward_encoded_sequences[i]==km_token_id2)] for i, score in enumerate(reward_scores)]
            reward = reward*2-1
            return reward
        _rewards = []
        offset = 0
        for i in range(len(critic_packed_seq_lens)):
            _rewards.append(process_reward(rewards[:,offset:offset+critic_packed_seq_lens[i]]))
            offset += critic_packed_seq_lens[i]

        rewards = [r.item() for r in _rewards]

        return rewards


    @torch.no_grad()
    def make_experience_orm(self, samples: Samples) -> Experience:
        km_token = 'ки'
        km_token_id1 = 12902
        km_token_id2 = 1107
        good_token_id = 648
        bad_token_id = 387
        candidate_tokens = [good_token_id, bad_token_id]

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
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        # 根据packed_seq_lens 分割成多段，进行decode
        sequences_split = []
        offset = 0
        tokens_list = sequences_cpu.tolist()[0]
        for length in packed_seq_lens:
            # 去掉eottoken
            sequences_split.append(tokens_list[offset : offset + length-1])
            offset += length
        decoded_sequences = self.tokenizer.batch_decode(sequences_split, skip_special_tokens=False)
        # decoded_sequences = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
        # decoded_sequences = [seq.replace(km_token, ' ') + km_token for seq in decoded_sequences]
        # critic_inputs = self.tokenize_fn(decoded_sequences, self.prompt_max_len*10, device='cuda', tokenizer=self.critic_tokenizer, padding=False)
        # critic_encoded_sequences = critic_inputs['input_ids']
        # critic_attention_mask = critic_inputs['attention_mask']
        # critic_packed_seq_lens = []
        # # sequences 是二维的，获取每行长度，作为critic_packed_seq_lens
        # for seq in critic_encoded_sequences:
        #     critic_packed_seq_lens.append(len(seq))
        # # 展平critic_encoded_sequences, critic_encoded_mask为一维
        # _critic_encoded_sequences = []
        # for seq in critic_encoded_sequences:
        #     _critic_encoded_sequences.extend(seq)
        # critic_encoded_sequences = torch.tensor(_critic_encoded_sequences).unsqueeze(0)
        # _critic_attention_mask = []
        # # TODO: 不太对, 确认一下
        # for i, mask in enumerate(critic_attention_mask):
        #     # mask*(i+1) 
        #     _critic_attention_mask.extend([v*(i+1) for v in mask])
        # critic_attention_mask = torch.tensor(_critic_attention_mask).unsqueeze(0)
        r_refs = []
        # support remote RM API with ray
        # if not self.remote_rm_url:
        #     for rm in self.reward_model:
        #         r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        # else:
        #     # remote RM
        #     for rm in self.remote_rm_url:
        #         if not self.packing_samples:
        #             queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
        #             r = remote_rm_fn_ray.remote(rm, queries=queries)
        #             r_refs.append(r)
        #         else:
        #             sequences_list = []
        #             offset = 0
        #             tokens_list = sequences_cpu.tolist()[0]
        #             for length in packed_seq_lens:
        #                 sequences_list.append(tokens_list[offset : offset + length])
        #                 offset += length
        #             queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
        #             r = remote_rm_fn_ray.remote(rm, queries=queries)
        #             r_refs.append(r)
        # get prm as orm
        rewards_ref = self.reward_model[0].get_orm_from_prm.remote(decoded_sequences)
        # list[float]
        rewards = ray.get(rewards_ref)
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        r = torch.tensor(rewards).reshape(-1, 1).to(device)

        # get prm as orm

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start
        base_action_log_probs, value = ref_values[0], ref_values[1]

        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()
        
        # if self.advantage_estimator == 'group_norm':
        #     kl = None
        # else:
        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1) if kl is not None else None
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions) if kl is not None else None
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device) if kl is not None else None
        info = {
            # "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "token_counter": torch.tensor(samples.token_counters, dtype=torch.float32),
            "call_counter": torch.tensor(samples.call_counters, dtype=torch.float32),
        }
        if kl is not None:
            info['kl'] = kl_mean

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time
        experience = Experience(
            sequences,
            action_log_probs,
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
    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        if self.strategy.args.mcts_mode:
            # return self.make_experience_prm(samples)
            return self.make_experience_mcts(samples)
        else:
            if self.use_prm:
                return self.make_experience_prm(samples)
            else:
                return self.make_experience_orm(samples)
    

        
    
    # @torch.no_grad()
    # def make_experience(self, samples: Samples) -> Experience:
    #     """
    #     Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
    #     """
    #     self.actor.eval()
    #     device = torch.cuda.current_device()

    #     # extract values from samples
    #     sequences = samples.sequences
    #     attention_mask = samples.attention_mask
    #     action_mask = samples.action_mask
    #     num_actions = samples.num_actions
    #     packed_seq_lens = samples.packed_seq_lens

    #     start = time.time()
    #     sequences_cpu, attention_mask_cpu = (
    #         sequences.to("cpu"),
    #         attention_mask.to("cpu"),
    #     )

    #     # init log probs
    #     base_action_log_probs_ref = self.initial_model.forward.remote(
    #         sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
    #     )

    #     # values
    #     if self.critic is not None:
    #         value_ref = self.critic.forward.remote(
    #             sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
    #         )
    #         # avoid CUDA OOM when colocate models
    #         if self.strategy.args.colocate_critic_reward:
    #             ray.get([value_ref])
    #             ray.get([self.critic.empty_cache.remote()])
    #     else:
    #         value_ref = ray.put(None)

    #     if self.strategy.args.colocate_actor_ref:
    #         ray.get([base_action_log_probs_ref])
    #         ray.get([self.initial_model.empty_cache.remote()])

    #     # rewards
    #     r_refs = []
    #     # support remote RM API with ray
    #     if not self.remote_rm_url:
    #         for rm in self.reward_model:
    #             r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
    #     else:
    #         # remote RM
    #         for rm in self.remote_rm_url:
    #             if not self.packing_samples:
    #                 queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
    #                 r = remote_rm_fn_ray.remote(rm, queries=queries)
    #                 r_refs.append(r)
    #             else:
    #                 sequences_list = []
    #                 offset = 0
    #                 tokens_list = sequences_cpu.tolist()[0]
    #                 for length in packed_seq_lens:
    #                     sequences_list.append(tokens_list[offset : offset + length])
    #                     offset += length
    #                 queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
    #                 r = remote_rm_fn_ray.remote(rm, queries=queries)
    #                 r_refs.append(r)

    #     # log probs
    #     action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
    #     actor_value_rm_time = time.time() - start

    #     # wait initial/critic/reward model done
    #     start = time.time()
    #     ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
    #     wait_time = time.time() - start

    #     base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
    #     base_action_log_probs = base_action_log_probs.to(device)
    #     if value is not None:
    #         value = value.to(device)
    #     rewards = [r.to(device) for r in rewards]
    #     r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

    #     # avoid CUDA OOM when colocate models
    #     if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
    #         ray.get([self.reward_model[0].empty_cache.remote()])

    #     if self.strategy.args.colocate_actor_ref:
    #         torch.cuda.empty_cache()

    #     kl = compute_approx_kl(
    #         action_log_probs,
    #         base_action_log_probs,
    #         action_mask=action_mask,
    #         use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
    #     )

    #     if not self.packing_samples:
    #         kl_mean = masked_mean(kl, action_mask, dim=-1)
    #     else:
    #         # convert tensor into list of tensors so that it's easier to manipulate
    #         # within dataset.
    #         sequences = unpacking_samples(sequences, packed_seq_lens)
    #         attention_mask = None
    #         action_log_probs = unpacking_samples(action_log_probs, num_actions)
    #         if value is not None:
    #             value = unpacking_samples(value, num_actions)

    #         kl = unpacking_samples(kl, num_actions)
    #         kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

    #     info = {
    #         "kl": kl_mean,
    #         "reward": r,
    #         "response_length": samples.response_length,
    #         "total_length": samples.total_length,
    #         "num_actions": num_actions,
    #     }

    #     if self.strategy.args.perf:
    #         self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
    #         self.perf_stats["wait_time"] += wait_time

    #     experience = Experience(
    #         sequences,
    #         action_log_probs,
    #         value,
    #         None,
    #         None,
    #         attention_mask,
    #         action_mask,
    #         info,
    #         kl,
    #     )

    #     self.actor.train()  # reset model state
    #     return experience
    
    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

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
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
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

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

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
                token_counters = []
                call_counters = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    token_counters.append(output_len)
                    call_counters.append(1)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

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
                        token_counters=token_counters,
                        call_counters=call_counters
                    )
                )
        return samples_list

    def _mcts_vllm(self, all_prompts: List[str], all_solutions: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        #
        # # round-robin load balance
        # rank = torch.distributed.get_rank()
        # world_size = torch.distributed.get_world_size()

        # # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        # if len(self.vllm_engines) <= world_size:
        #     llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        # else:
        #     llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_gt_solutions = sum([[solution] * args.n_samples_per_prompt for solution in all_solutions], [])
        samples_list = []
        # list[list[str]:num_rollouts]
        all_model_solutions = []
        all_node_solutions = []
        all_token_counter = []
        all_call_counter = []
        for user_question, gt_solution in zip(all_prompts, all_gt_solutions):
            # 计数器归零
            self.generator.io.call_counter = 0
            self.generator.io.token_counter = 0
            gt_answer = self.evaluator.extract_answer_from_gold_solution(gt_solution)
            #! build an MCTS searcher
            mcts_searcher = MCTS_Searcher(
                exploration_weight=args.mcts_exploration_weight,
                weight_scheduler=args.mcts_weight_scheduler,
                num_rollouts=args.num_rollouts,
                discount=args.mcts_discount_factor,
                verbose=args.verbose,
                mcts_use_prm=args.mcts_use_prm
            )

            #! build the MCTS tree
            if self.strategy.args.mcts_mode == 'only_ost':
                root_node = OstOrAnswerNode(
                    parent=None,
                    depth=0,
                    node_type=Node_Type.USER_QUESTION,
                    verbose=args.verbose,
                    generator=self.generator,
                    disable_a5=args.disable_a5,
                    user_question=user_question,
                    expected_answer=gt_answer,
                    max_depth_allowed=args.max_depth_allowed,
                    disable_a1=args.disable_a1,
                    enable_potential_score=args.enable_potential_score,
                )
            elif self.strategy.args.mcts_mode == 'full_actions':
                root_node = Reasoning_MCTS_Node(
                    parent=None,
                    depth=0,
                    node_type=Node_Type.USER_QUESTION,
                    verbose=args.verbose,
                    generator=self.generator,
                    disable_a5=args.disable_a5,
                    user_question=user_question,
                    expected_answer=gt_answer,
                    max_depth_allowed=args.max_depth_allowed,
                    disable_a1=args.disable_a1,
                    enable_potential_score=args.enable_potential_score,
                )

            model_solutions = []
            solution_nodes = []
            model_all_solutions = []
            model_rollout_nodes = []
            for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
                rollout_node = mcts_searcher.do_rollout(root_node, i)
                model_rollout_nodes.append(rollout_node)
                _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
                    root_node, self.generator.evaluator, enable_potential_score=args.enable_potential_score
                )
                
                # model_solutions.append(best_solution)
                # model_all_solutions.append(all_solutions)
                if args.save_tree:
                    with open(
                        os.path.join(
                            args.answer_sheets_dir,
                            f"Question {user_question[:20]} - Rollout {i}.tree",
                        ),
                        "w",
                    ) as f:
                        print_tree_from_root(
                            mcts_searcher=mcts_searcher,
                            rollout_id=i,
                            root_node=root_node,
                            chosen_node=chosen_node,
                            file=f,
                        )
            # only response
            if best_solution is None:
                best_solution = self.vllm_generator.generate_outputs(user_question, sampling_params)[0].outputs[0].text
                chosen_node = None
            model_solutions.append(best_solution)
            solution_nodes.append(chosen_node)
            js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
            with open(os.path.join(args.answer_sheets_dir, "Question "+re.sub(r'[^a-zA-Z0-9+\-$()\[\]]', '_', user_question[:25])+" - Final Solutions.json"), "w") as f:
                json.dump(js, f)

            js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
            with open(os.path.join(args.answer_sheets_dir, "Question "+re.sub(r'[^a-zA-Z0-9+\-$()\[\]]', '_', user_question[:25])+" - Rollout Solutions.json"), "w") as f:
                json.dump(js2, f)

            all_model_solutions.append(model_solutions)
            all_node_solutions.append(solution_nodes)
            all_token_counter.append(self.generator.io.token_counter)
            all_call_counter.append(self.generator.io.call_counter)
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            model_solutions = all_model_solutions[i:i+args.micro_rollout_batch_size]
            node_solutions = all_node_solutions[i:i+args.micro_rollout_batch_size]
            token_counters = all_token_counter[i:i+args.micro_rollout_batch_size]
            call_counters = all_call_counter[i:i+args.micro_rollout_batch_size]

            _model_solutions = []
            solution_rewards = []
            for i, solutions in enumerate(model_solutions):
                nodes = node_solutions[i]
                for j, solution in enumerate(solutions):
                    _model_solutions.append(solution)
                    if nodes[j] is not None:
                        solution_rewards.append(nodes[j].node_value)
                    else:
                        solution_rewards.append(0)
            model_solutions = _model_solutions

            if not self.packing_samples:
                raise NotImplementedError
            else:
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                prompt_token_id = self.tokenizer.encode(all_prompts[i], add_special_tokens=False)
                input_len = len(prompt_token_id)
                resps_token_id = [
                    self.tokenizer.encode(solution, add_special_tokens=False) 
                    for solution in model_solutions
                ]
                
                for i, resp_token_id in enumerate(resps_token_id):
                    output_len = len(resp_token_id)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(prompt_token_id + resp_token_id)
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                solution_rewards = torch.tensor(solution_rewards, device="cuda", dtype=torch.float).reshape(-1, 1)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        rewards=solution_rewards,
                        call_counters=call_counters,
                        token_counters=token_counters,
                    )
                )
        return samples_list



        
        # Expand prompt list based on the number of samples per prompt
        
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompts[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
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

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

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
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

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
                    )
                )
        return samples_list
    
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
