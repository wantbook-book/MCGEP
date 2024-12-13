import logging
import os
import socket
from typing import Callable, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import DeepspeedStrategy, get_tokenizer


class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
            vocab_size=strategy.args.vocab_size,
        )

        self.tokenizer = get_tokenizer(
            pretrain, model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def get_tokenizer(self):
        return self.tokenizer
    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, packed_seq_lens=None
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device), packed_seq_lens=packed_seq_lens)
        return reward.to("cpu")

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
    
    # TODO: for mistral prm
    # reward_encoded_inputs: reward_tokenizer(km_join_p_responses), km_join_p_responses只有一个km
    def get_orm_from_prm(self, decoded_sequences: list[str])->list[float]:
        res = self.get_prm_res(decoded_sequences, return_seqs_and_masks=False)
        rewards = res["rewards"]
        packed_seq_lens = res["packed_seq_lens"]
        _rewards = []
        offset = 0
        for i, seq_len in enumerate(packed_seq_lens):
            idx = offset + seq_len - 1
            _rewards.append(rewards[0, idx].cpu().item())
            offset += seq_len
        return _rewards
        # rewards_neq0 = rewards[rewards!=0]
        # if rewards_neq0.numel() != 0:
        #     return [rewards_neq0[-1].item()]
        # else:
        #     return [0]

    
    # TODO: for mistral prm
    # reward_encoded_inputs: reward_tokenizer(km_join_p_responses)
    def get_prm_res(self, decoded_sequences: list[str], return_seqs_and_masks: bool=False)->dict:
        km_token = 'ки'
        km_token_id1 = 12902
        km_token_id2 = 1107
        good_token_id = 648
        bad_token_id = 387
        sep_token1 = '\n'
        sep_token2 = '\n\n'
        candidate_tokens = [good_token_id, bad_token_id]
        prompt_max_len = self.strategy.args.prompt_max_len

        # try_split = decoded_sequences[0].split(sep_token1)
        # if '' in try_split:
        #     sep_token = sep_token2
        # else:
        #     sep_token = sep_token1
        sep_token = sep_token1
        split_actor_p_responses = [resp.split(sep_token) for resp in decoded_sequences if resp != '']

        # 给每个response加上sep_token，过滤掉空字符串
        # 去掉原来里面存在的km token
        split_actor_p_responses = [[resp.replace(km_token, '')+sep_token for resp in split_resp if resp.strip()] for split_resp in split_actor_p_responses]
        for split_resp in split_actor_p_responses:
            # 去掉最后一个换行符\n
            split_resp[-1] = split_resp[-1][:-1]
        
        # TODO: 检查下有没有填充
        split_actor_seqs = [[self.tokenize_fn(resp, prompt_max_len, device='cuda') for resp in split_resp] for split_resp in split_actor_p_responses]
        # 元素为torch.tensor, (1, len)
        #[
        # [
        #   [1,2,2],[2,3,3,5]
        # ]
        #]
        split_actor_seqs = [[item['input_ids'][0] for item in split_seq] for split_seq in split_actor_seqs ]
        # 用于确定每个step的reward位置
        # [
        #   [3, 4]
        # ]
        split_actor_lens = [[len(seq) for seq in split_seq] for split_seq in split_actor_seqs]

        # [
        #    [1,2,2,2,3,3,5]
        #]
        concatenated_actor_seqs = []
        for split_actor_seq in split_actor_seqs:
            # 展平二维列表
            flat_list = [item for sublist in split_actor_seq for item in sublist]
            concatenated_actor_seqs.append(flat_list)
        # [7, ]
        packed_seq_lens = [len(seq) for seq in concatenated_actor_seqs]
        sequences = []
        attention_mask = []
        for i, seq in enumerate(concatenated_actor_seqs):
            sequences.extend(seq)
            attention_mask.extend([1]*len(seq))
        # (1, all_seq_len_sum)
        sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
        sequences_cpu = sequences.to("cpu")
        attention_mask_cpu = attention_mask.to("cpu")

        km_join_p_responses = [km_token.join(split_resp)+km_token for split_resp in split_actor_p_responses]
        reward_encoded_inputs = self.tokenize_fn(km_join_p_responses, prompt_max_len*10, device='cuda', padding=False)
        reward_encoded_sequences = reward_encoded_inputs['input_ids']
        reward_encoded_attention_mask = reward_encoded_inputs['attention_mask']
        reward_packed_seq_lens = []
        # sequences 是二维的，获取每行长度，作为critic_packed_seq_lens
        for seq in reward_encoded_sequences:
            reward_packed_seq_lens.append(len(seq))
        
        # 展平critic_encoded_sequences, critic_encoded_mask为一维
        _reward_encoded_sequences = []
        for seq in reward_encoded_sequences:
            _reward_encoded_sequences.extend(seq)
        reward_encoded_sequences = torch.tensor(_reward_encoded_sequences).unsqueeze(0)
        _reward_attention_mask = []
        for mask in reward_encoded_attention_mask:
            _reward_attention_mask.extend([v*(i+1) for v in mask])
        reward_encoded_attention_mask = torch.tensor(_reward_attention_mask).unsqueeze(0)

        reward_logits = self.forward(reward_encoded_sequences, reward_encoded_attention_mask, packed_seq_lens=reward_packed_seq_lens)
        # (1, batch_size*seq_step_each, vocab_size)
        # reward_logits = ray.get(r_ref)

        reward_logits = reward_logits[..., candidate_tokens]
        reward_scores = reward_logits.softmax(dim=-1)[:,:,0]
        # TODO: 需要挑选个好的
        # 大于0.7认为正确，小于0.43认为错误
        reward_scores_gt_07_mask = reward_scores > 0.7
        rewrad_scores_lt_043_mask = reward_scores < 0.43
        reward_scores = 2/(0.7-0.43)*(reward_scores-0.43) - 1
        reward_scores[reward_scores_gt_07_mask] = 1
        reward_scores[rewrad_scores_lt_043_mask] = -1
        step_rewards = reward_scores[(reward_encoded_sequences==km_token_id1) | (reward_encoded_sequences==km_token_id2)]

        # 转成二维
        step_rewards_2d = []
        rewards = torch.zeros_like(sequences, device='cuda', dtype=torch.bfloat16)
        start = 0
        step_i = 0
        for one_split_lens in split_actor_lens:
            step_reward = []
            for split_len in one_split_lens:
                idx = start + split_len - 1
                try:
                    step_reward.append(step_rewards[step_i].item())
                    rewards[0, idx] = step_rewards[step_i]
                except Exception as e:
                    breakpoint()
                start += split_len
                step_i += 1
            step_rewards_2d.append(step_reward)
        # 如果len(reward)!=0，否则avg=0
        avg_step_rewards = []
        for reward in step_rewards_2d:
            if len(reward):
                avg_step_rewards.append(sum(reward)/len(reward))
            else:
                avg_step_rewards.append(0)
        res = {
            "avg_step_rewards": avg_step_rewards,
            "rewards": rewards,
            "packed_seq_lens": packed_seq_lens,
        }

        if return_seqs_and_masks:
            res["sequences"] = sequences
            res["attention_mask"] = attention_mask
        return res

    # TODO: for mistral prm
    # reward_encoded_inputs: reward_tokenizer(km_join_p_responses)
    def get_prm_q_a_res(self, question:str, responses: list[str], return_seqs_and_masks: bool=False)->dict:
        km_token = 'ки'
        km_token_id1 = 12902
        km_token_id2 = 1107
        good_token_id = 648
        bad_token_id = 387
        sep_token1 = '\n'
        sep_token2 = '\n\n'
        candidate_tokens = [good_token_id, bad_token_id]
        prompt_max_len = self.strategy.args.prompt_max_len

        # try_split = decoded_sequences[0].split(sep_token1)
        # if '' in try_split:
        #     sep_token = sep_token2
        # else:
        #     sep_token = sep_token1
        sep_token = sep_token1
        split_actor_responses = [resp.split(sep_token) for resp in responses if resp != '']

        # 给每个response加上sep_token，过滤掉空字符串
        # 去掉原来里面存在的km token
        split_actor_responses = [[resp.replace(km_token, '')+sep_token for resp in split_resp if resp.strip()] for split_resp in split_actor_responses]
        for split_resp in split_actor_responses:
            # 去掉最后一个换行符\n
            split_resp[-1] = split_resp[-1][:-1]
        
        split_actor_q_responses = [[question+sep_token]+split_resp for split_resp in split_actor_responses]
        
        # TODO: 检查下有没有填充
        split_actor_seqs = [[self.tokenize_fn(resp, prompt_max_len, device='cuda') for resp in split_resp] for split_resp in split_actor_q_responses]
        # 元素为torch.tensor, (1, len)
        #[
        # [
        #   [1,2,2],[2,3,3,5]
        # ]
        #]
        split_actor_seqs = [[item['input_ids'][0] for item in split_seq] for split_seq in split_actor_seqs ]
        # 用于确定每个step的reward位置
        # [
        #   [3, 4]
        # ]
        split_actor_lens = [[len(seq) for seq in split_seq] for split_seq in split_actor_seqs]

        # [
        #    [1,2,2,2,3,3,5]
        #]
        concatenated_actor_seqs = []
        for split_actor_seq in split_actor_seqs:
            # 展平二维列表
            flat_list = [item for sublist in split_actor_seq for item in sublist]
            concatenated_actor_seqs.append(flat_list)
        # [7, ]
        packed_seq_lens = [len(seq) for seq in concatenated_actor_seqs]
        sequences = []
        attention_mask = []
        for i, seq in enumerate(concatenated_actor_seqs):
            sequences.extend(seq)
            attention_mask.extend([i+1]*len(seq))
        # (1, all_seq_len_sum)
        sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
        sequences_cpu = sequences.to("cpu")
        attention_mask_cpu = attention_mask.to("cpu")

        km_join_p_responses = [km_token.join(split_resp)+km_token for split_resp in split_actor_q_responses]
        reward_encoded_inputs = self.tokenize_fn(km_join_p_responses, prompt_max_len*10, device='cuda', padding=False)
        reward_encoded_sequences = reward_encoded_inputs['input_ids']
        reward_encoded_attention_mask = reward_encoded_inputs['attention_mask']
        reward_packed_seq_lens = []
        # sequences 是二维的，获取每行长度，作为critic_packed_seq_lens
        for seq in reward_encoded_sequences:
            reward_packed_seq_lens.append(len(seq))
        
        # 展平critic_encoded_sequences, critic_encoded_mask为一维
        _reward_encoded_sequences = []
        for seq in reward_encoded_sequences:
            _reward_encoded_sequences.extend(seq)
        reward_encoded_sequences = torch.tensor(_reward_encoded_sequences).unsqueeze(0)
        _reward_attention_mask = []
        for i, mask in enumerate(reward_encoded_attention_mask):
            _reward_attention_mask.extend([v*(i+1) for v in mask])
        reward_encoded_attention_mask = torch.tensor(_reward_attention_mask).unsqueeze(0)

        reward_logits = self.forward(reward_encoded_sequences, reward_encoded_attention_mask, packed_seq_lens=reward_packed_seq_lens)
        # (1, batch_size*seq_step_each, vocab_size)
        # reward_logits = ray.get(r_ref)

        reward_logits = reward_logits[..., candidate_tokens]
        reward_scores = reward_logits.softmax(dim=-1)[:,:,0]
        # TODO: 需要挑选个好的
        # 大于0.7认为正确，小于0.43认为错误
        reward_scores_gt_07_mask = reward_scores > 0.7
        rewrad_scores_lt_043_mask = reward_scores < 0.43
        reward_scores = 2/(0.7-0.43)*(reward_scores-0.43) - 1
        reward_scores[reward_scores_gt_07_mask] = 1
        reward_scores[rewrad_scores_lt_043_mask] = -1
        step_rewards = reward_scores[(reward_encoded_sequences==km_token_id1) | (reward_encoded_sequences==km_token_id2)]

        # 转成二维
        step_rewards_2d = []
        rewards = torch.zeros_like(sequences, device='cuda', dtype=torch.bfloat16)
        start = 0
        step_i = 0
        for one_split_lens in split_actor_lens:
            step_reward = []
            for split_len in one_split_lens:
                idx = start + split_len - 1
                step_reward.append(step_rewards[step_i].item())
                rewards[0, idx] = step_rewards[step_i]
                start += split_len
                step_i += 1
            step_rewards_2d.append(step_reward)
        # 如果len(reward)!=0，否则avg=0
        # avg_step_rewards = []
        # for reward in step_rewards_2d:
        #     if len(reward):
        #         avg_step_rewards.append(sum(reward)/len(reward))
        #     else:
        #         avg_step_rewards.append(0)
        # res = {
        #     "avg_step_rewards": avg_step_rewards,
        #     "rewards": rewards,
        #     "packed_seq_lens": packed_seq_lens,
        # }

        # if return_seqs_and_masks:
        #     res["sequences"] = sequences
        #     res["attention_mask"] = attention_mask
        res = {
            "step_rewards_2d": step_rewards_2d,
            "split_q_responses": split_actor_q_responses
        }
        return res

    def get_orm(self):
        pass


    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BasePPORole],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
            ]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        critic_model_group: "PPORayActorGroup",
        initial_model_group: "PPORayActorGroup",
        reward_model_groups: List["PPORayActorGroup"],
        remote_rm_urls: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
    ):
        """Train actor model.

        Args:
            critic_model_group (PPORayActorGroup): critic model group.
            initial_model_group (PPORayActorGroup): reference model group.
            reward_model_groups (PPORayActorGroup): reward model groups.
            remote_rm_urls: remote RM APIs.
            reward_fn: reward calculate function, must be specified if using multiple reward models.
            vllm_engines: vllm engines for text generation, if not specified, generate text by actor model directly.

        Returns:
            List: list of remote object refs.
        """
        assert (
            (remote_rm_urls and len(remote_rm_urls) == 1)
            or (reward_model_groups and len(reward_model_groups) == 1)
            or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        critic_actors = critic_model_group._actor_handlers if critic_model_group else None
        initial_actors = initial_model_group._actor_handlers

        refs = []
        # TODO(wuxibin): actor model choose critic/reward/initial model in a
        # round robin fashion, implement more efficient dispatching strategy.
        for i, actor in enumerate(self._actor_handlers):
            critic_actor = critic_actors[i % len(critic_actors)] if critic_actors else None
            initial_actor = initial_actors[i % len(initial_actors)]

            reward_actors = []
            if not remote_rm_urls:
                for reward_model_group in reward_model_groups:
                    actors = reward_model_group._actor_handlers
                    reward_actors.append(actors[i % len(actors)])
            refs.append(
                actor.fit.remote(
                    critic_tokenizer=ray.get(reward_model_group._actor_handlers[0].get_tokenizer.remote()),
                    critic_model=critic_actor,
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_urls,
                    reward_fn=reward_fn,
                    vllm_engines=vllm_engines,
                    # whether this actor should triger corresponding critic model training
                    critic_train_remote=(i < len(critic_actors)) if critic_actor else None,
                )
            )

        return refs

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs
