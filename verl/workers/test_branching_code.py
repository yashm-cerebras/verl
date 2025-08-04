
class TreeNode:  
    def __init__(self, node_id: int, parent_id: int = None):  
        self.node_id = node_id  
        self.parent_id = parent_id  
        self.children = []  # List of child TreeNode objects  
        self.sequence = None  # Sequence of tokens for this node  
        self.reward = None  # Reward value for this node, tensor rewards
        self.is_leaf = True  
        self.text_until_this_node = ""  # Text generated up to this node
        self.generated_tokens = []  # Tokens generated at this node


class TreeRollout(BaseRollout):  
    def build_dataproto_for_verl(
        self,
        token_ids: List[int], # TODO: Check if this data type is correct
        pad_token_id: int,
        max_prompt_length: int,
        eos_token_id: int,
    ) -> DataProto:
        # Set tokenizer to Verl's padding style
        self.tokenizer.padding_side = "left"

        # Pad using tokenizer (returns tensors)
        padded = self.tokenizer.pad(
            [{"input_ids": token_ids}],
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = padded["input_ids"]  # shape: (1, max_prompt_length)
        attention_mask = padded["attention_mask"]
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)  # (1, max_prompt_length)

        batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=1)

        non_tensor_batch = {
            "raw_prompt_ids": np.array([token_ids], dtype=object)  # left-padding removed
        }

        meta_info = {
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "do_sample": True,
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def generate_tokens_from_verl(
        self,
        input_tokens: List[int], # TODO: Check if this data type is correct
        num_tokens: int, # Number of tokens to generate
        **kwargs,
    ) -> List[int]:
        dp = self.build_dataproto_for_verl(
            token_ids=input_tokens,
            pad_token_id=self.model.config.pad_token_id,
            max_prompt_length=self.model.config.prompt_length,
            eos_token_id=self.model.config.eos_token_id,
        )

        output = self.model.generate_sequences(dp, **kwargs)

        tokens = output.batch["responses"][0].tolist()
        return tokens[:num_tokens]


    def extract_leaf_paths(self, tree_data: Dict[int, TreeNode]) -> List[List[TreeNode]]:
        """Return a list of paths from root to each leaf node (as TreeNode objects)."""
        leaf_paths = []

        for node in tree_data.values():
            if node.is_leaf:
                path = []
                current = node
                while current is not None:
                    path.append(current)
                    current = tree_data.get(current.parent_id)
                leaf_paths.append(list(reversed(path)))  # root to leaf

        return leaf_paths

    def create_tree(
        self,
        prompts: DataProto,
        branching_config: Callable[[int], int],
        tokens_per_step_config: Callable[[int], int],
        **kwargs
    ) -> Dict[int, TreeNode]:

        self.tree_nodes = {}
        self.step_groups = {}

        # Extract root prompt (unpadded, already left-stripped by verl)
        root_input_ids = prompts.non_tensor_batch["raw_prompt_ids"][0]
        root_node = TreeNode(node_id=0)
        root_node.sequence = list(root_input_ids)
        root_node.depth = 0
        self.tree_nodes[0] = root_node
        self.step_groups[0] = [0]

        queue = deque([root_node])
        next_node_id = 1

        while queue:
            node = queue.popleft()

            node.depth = 0 if node.parent_id is None else self.tree_nodes[node.parent_id].depth + 1
            if node.depth >= self.max_depth:
                continue

            self.step_groups.setdefault(node.depth, []).append(node.node_id)

            tokens_to_generate = tokens_per_step_config(node.depth)
            generated_tokens = self.generate_tokens_from_verl(node.sequence, tokens_to_generate, **kwargs)
            new_sequence = node.sequence + generated_tokens

            node.generated_tokens = generated_tokens
            node.sequence = new_sequence

            if self.branching_strategy.should_branch(len(new_sequence), context={"node": node}):
                num_children = branching_config(node.depth)
                for _ in range(num_children):

                    child = TreeNode(node_id=next_node_id, parent_id=node.node_id)
                    child.sequence = new_sequence 
                    child.depth = node.depth + 1

                    node.children.append(child)
                    node.is_leaf = False # Mark parent as non-leaf

                    self.tree_nodes[next_node_id] = child
                    queue.append(child)
                    next_node_id += 1

        return self.tree_nodes
