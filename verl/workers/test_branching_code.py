from collections import deque
from typing import List, Dict, Callable, Tuple
import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import DictConfig
from verl import DataProto
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask


class DefaultBranchingStrategy:
    # it just branches at a default token length every node (k tokens each node)
    def should_branch(self, _seq_len: int, context: dict) -> bool:
        return True


class TreeNode:
    def __init__(self, node_id: int, parent_id: int = None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children = []
        self.sequence = None
        self.reward = None
        self.is_leaf = True
        self.generated_tokens = []
        self.depth = 0

class TreeRollout(vLLMRollout):
    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        branching_strategy = None,
        max_depth: int = 3,
        **kwargs,
    ):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer              # base class doesn’t store this
        if self.branching_strategy is None:
            self.branching_strategy = DefaultBranchingStrategy()
        self.max_depth = max_depth
        self.tree_nodes: Dict[int, TreeNode] = {}
        self.step_groups: Dict[int, List[int]] = {}
        self.max_response_length = config.response_length  # default max response length
      

    # Build a single-item DataProto using tokenizer-based left padding
    def _build_dataproto(self, token_ids: List[int]) -> DataProto:
        self.tokenizer.padding_side = "left"
        padded = self.tokenizer.pad(
            [{"input_ids": token_ids}],
            padding="max_length",
            max_length=self.config.prompt_length,  # from vLLMRollout config
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = padded["input_ids"]                  # (1, prompt_len)
        attention_mask = padded["attention_mask"]        # (1, prompt_len)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)  # (1, prompt_len)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=1,
        )
        non_tensor_batch = {"raw_prompt_ids": np.array([token_ids], dtype=object)}
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def generate_tokens_from_verl(self, input_tokens: List[int], num_tokens: int, **kwargs) -> List[int]:
        dp = self._build_dataproto(input_tokens)
        out = self.generate_sequences(dp, **kwargs)  # method from vLLMRollout
        resp = out.batch["responses"][0].tolist()
        return resp[:num_tokens]


    def build_leaf_reward_batch(self) -> Tuple[DataProto, List[int]]:
        """
        Build a DataProto from all leaf (root→leaf) sequences, matching the shape/logic
        of vLLMRollout.generate_sequences:
        - prompts:        [B, prompt_length]
        - responses:      [B, response_length]
        - input_ids:      [B, prompt_length + response_length]  (concat of prompts & responses)
        - attention_mask: [B, prompt_length + response_length]
        - position_ids:   [B, prompt_length + response_length]

        Returns: (DataProto, leaf_node_ids)
        """
        leaves: List[TreeNode] = [n for n in self.tree_nodes.values() if n.is_leaf]
        leaf_ids: List[int] = [n.node_id for n in leaves]
        seqs: List[List[int]] = [n.sequence for n in leaves]

        prompt_len = int(self.config.prompt_length)
        resp_len = int(self.config.response_length)
        total_len = prompt_len + resp_len

        pad_id = self.tokenizer.pad_token_id
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        if not hasattr(self, "prompt_len"):
            raise RuntimeError("TreeRollout: self.prompt_len is required. Set it in create_tree().")

        # left-padding raw prompt tokens to prompt_len 
        raw_prompts = [seq[: self.prompt_len] for seq in seqs]
        self.tokenizer.padding_side = "left"
        prompts_pad = self.tokenizer.pad(
            [{"input_ids": rp} for rp in raw_prompts],
            padding="max_length",
            max_length=prompt_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        idx = prompts_pad["input_ids"]                 # [B, prompt_len]
        prompt_attn = prompts_pad["attention_mask"]    # [B, prompt_len]

        #  Build responses by right-padding/truncating to resp_len 
        responses_list = [seq[self.prompt_len:] for seq in seqs]
        response = pad_2d_list_to_length(responses_list, pad_token_id=pad_id, max_length=resp_len)

        # following gen_sequences setup
        seq = torch.cat([idx, response], dim=-1)       # [B, total_len]

    
        B = idx.size(0)
        prompt_pos = torch.arange(prompt_len, device=idx.device).unsqueeze(0).expand(B, -1)
        delta = torch.arange(1, resp_len + 1, device=idx.device).unsqueeze(0).expand(B, -1)
        response_pos = prompt_pos[:, -1:].to(delta.dtype) + delta
        position_ids = torch.cat([prompt_pos, response_pos], dim=-1)  # [B, total_len]

        #  attention_mask: concat prompt mask with response mask (Verl util) 
        if eos_id is None:
            response_mask = (response != pad_id).to(prompt_attn.dtype)
        else:
            response_mask = get_response_mask(response_id=response, eos_token=eos_id, dtype=prompt_attn.dtype)
        attention_mask = torch.cat([prompt_attn, response_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=B,
        )

        non_tensor_batch = {
            "uid": np.array([str(leaf_id) for leaf_id in leaf_ids], dtype=object)
        }

        self.leaf_id_to_batch_idx = {leaf_id: i for i, leaf_id in enumerate(leaf_ids)}
        self.batch_id_to_leaf_id = {i: leaf_id for i, leaf_id in enumerate(leaf_ids)}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch), leaf_ids

    def extract_leaf_paths(self, tree_data: Dict[int, TreeNode]) -> List[List[TreeNode]]:
        paths = []
        for node in tree_data.values():
            if node.is_leaf:
                path = []
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = tree_data.get(cur.parent_id)
                paths.append(list(reversed(path)))
        return paths

    def create_tree(
        self,
        prompts: DataProto,
        branching_config: Callable[[int], int],
        tokens_per_step_config: Callable[[int], int],
        **kwargs,
    ) -> Dict[int, TreeNode]:
        self.tree_nodes = {}
        self.step_groups = {}

        # Use unpadded prompt tokens provided in DataProto
        root_input_ids = prompts.non_tensor_batch["raw_prompt_ids"][0]
        root = TreeNode(node_id=0)
        root.sequence = list(root_input_ids)
        root.depth = 0
        self.tree_nodes[0] = root
        self.step_groups[0] = [0]

       
        self.prompt_len = len(root_input_ids)


        q = deque([root])
        next_id = 1

        while q:
            node = q.popleft()
            node.depth = 0 if node.parent_id is None else self.tree_nodes[node.parent_id].depth + 1
            if node.depth >= self.max_depth:
                continue

            self.step_groups.setdefault(node.depth, []).append(node.node_id)

            to_gen = tokens_per_step_config(node.depth)
            gen_tokens = self.generate_tokens_from_verl(node.sequence, to_gen, **kwargs)
            node.generated_tokens = gen_tokens
            node.sequence = node.sequence + gen_tokens

            if len(node.sequence) >= self.max_response_length:
                # keep node as leaf (no children)
                continue

            if self.branching_strategy.should_branch(len(node.sequence), context={"node": node}):
                num_children = branching_config(node.depth)
                for _ in range(num_children):
                    child = TreeNode(node_id=next_id, parent_id=node.node_id)
                    child.sequence = node.sequence  # branch from the updated sequence
                    child.depth = node.depth + 1
                    node.children.append(child)
                    node.is_leaf = False
                    self.tree_nodes[next_id] = child
                    q.append(child)
                    next_id += 1

        return self.tree_nodes


