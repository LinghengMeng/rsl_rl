import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.modules.utils import get_activation
from rsl_rl.modules.transformer import Transformer
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.utils import squeeze_preserve_batch


class MultiHeadNetwork(Benchmarkable, nn.Module):
    recurrent_module_lstm = "LSTM"
    recurrent_module_transformer = "TF"

    recurrent_modules = {recurrent_module_lstm: nn.LSTM, recurrent_module_transformer: Transformer}

    def __init__(
        self,
        input_size: int,
        output_size: List[int],
        # activations: List[str] = ["relu", "relu", "relu", "tanh"],
        # hidden_dims: List[int] = [256, 256, 256],
        activations: List = ["relu", "relu", [["relu", "tanh"], ["relu", "tanh"], ["relu", "tanh"], ["relu", "tanh"]]],
        hidden_dims: List = [256, 256, [[256],[256],[256],[256]]],
        init_fade: bool = True,
        init_gain: float = 1.0,
        input_normalization: bool = False,
        recurrent: bool = False,
        recurrent_layers: int = 1,
        recurrent_module: str = recurrent_module_lstm,
        recurrent_tf_context_length: int = 64,
        recurrent_tf_head_count: int = 8,
    ) -> None:
        """

        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            activations (List[str]): The activation functions to use. If the network is recurrent, the first activation
                function is used for the output of the recurrent layer.
            hidden_dims (List[int]): The hidden dimensions. If the network is recurrent, the first hidden dimension is
                used for the recurrent layer.
            init_fade (bool): Whether to use the fade in initialization.
            init_gain (float): The gain to use for the initialization.
            input_normalization (bool): Whether to use input normalization.
            recurrent (bool): Whether to use a recurrent network.
            recurrent_layers (int): The number of recurrent layers (LSTM) / blocks (Transformer) to use.
            recurrent_module (str): The recurrent module to use. Must be one of Network.recurrent_modules.
            recurrent_tf_context_length (int): The context length of the Transformer.
            recurrent_tf_head_count (int): The head count of the Transformer.
        """
        # Check if hidden_dims matche activations
        assert len(hidden_dims) == len(activations)
        for dim, act in zip(hidden_dims, activations):
            # Check if the head hidden dim matches head activations
            if isinstance(dim, list):
                assert isinstance(act, list)    # make sure hidden dim and activation are all lists corresponding to head hidden dim and activation
                assert len(dim) == len(act)     # make sure the head number matches
                for head_hidden_dim, head_activation in zip(dim, act):
                    assert len(head_hidden_dim)+1 == len(head_activation)    # head hidden dim does not include output size, but head activation has output activation

        super().__init__()

        if input_normalization:
            self._normalization = EmpiricalNormalization(shape=(input_size,))
        else:
            self._normalization = nn.Identity()

        dims = [input_size] + hidden_dims 
        
        self._recurrent = recurrent
        self._recurrent_module = recurrent_module
        self.hidden_state = None
        self._last_hidden_state = None
        if self._recurrent:
            recurrent_kwargs = dict()

            if recurrent_module == self.recurrent_module_lstm:
                recurrent_kwargs["hidden_size"] = dims[1]
                recurrent_kwargs["input_size"] = dims[0]
                recurrent_kwargs["num_layers"] = recurrent_layers
            elif recurrent_module == self.recurrent_module_transformer:
                recurrent_kwargs["block_count"] = recurrent_layers
                recurrent_kwargs["context_length"] = recurrent_tf_context_length
                recurrent_kwargs["head_count"] = recurrent_tf_head_count
                recurrent_kwargs["hidden_size"] = dims[1]
                recurrent_kwargs["input_size"] = dims[0]
                recurrent_kwargs["output_size"] = dims[1]

            rnn = self.recurrent_modules[recurrent_module](**recurrent_kwargs)
            activation = get_activation(activations[0])
            dims = dims[1:]
            activations = activations[1:]

            self._features = nn.Sequential(rnn, activation)
        else:
            self._features = nn.Identity()

        # layers after the recurrent feature extraction 
        # print(dims)
        # print(activations)
        shared_pre_head_layers = []
        separate_multi_head_layers = {}
        for i in range(len(dims)-1):
            if not isinstance(dims[i+1], list):
                # Layers before multi-head
                layer = nn.Linear(dims[i], dims[i + 1])
                activation = get_activation(activations[i])
                shared_pre_head_layers.append(layer)
                shared_pre_head_layers.append(activation)
                # print(dims[i], dims[i + 1], activations[i])
            else:
                # Each head layers
                for head_i, (head_hidden_dim, head_activation) in enumerate(zip(dims[i+1], activations[i])):
                    # Note: head_activation includes the activation for the output layer, while head_hidden_dim does not include output layer size which is assumed to 1
                    separate_multi_head_layers[head_i] = []
                    head_dim = [dims[i]] + head_hidden_dim + [1]    # Add input and output to head_dim
                    for j in range(len(head_dim)-1):
                        layer = nn.Linear(head_dim[j], head_dim[j + 1])
                        activation = get_activation(head_activation[j])
                        separate_multi_head_layers[head_i].append(layer)
                        separate_multi_head_layers[head_i].append(activation)
                        # print('\t', head_dim[j], head_dim[j + 1], head_activation[j])
        # Put layers to container
        # self._shared_pre_head_layers = nn.Sequential(*shared_pre_head_layers)
        self._shared_pre_head_module_name = 'shared_pre_head_module'
        self.add_module(name=self._shared_pre_head_module_name, module=nn.Sequential(*shared_pre_head_layers))
        
        # 
        # self._separate_multi_head_layers = {head: nn.Sequential(*separate_multi_head_layers[head]) for head in separate_multi_head_layers}
        self._separate_multi_head_module_name_list = []
        for head_key, head_layers in separate_multi_head_layers.items():
            sub_module_name = 'head_{}'.format(head_key)
            self._separate_multi_head_module_name_list.append(sub_module_name)
            self.add_module(name=sub_module_name, module=nn.Sequential(*head_layers))
        print(self)  # Show the model
        
        # Initialize layers
        if len(self.get_submodule(self._shared_pre_head_module_name)) > 0:
            self._init(self._shared_pre_head_module_name, self._separate_multi_head_module_name_list, fade=init_fade, gain=init_gain) 

    @property
    def device(self):
        """Returns the device of the network."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The input data.
            hidden_state (Tuple[torch.Tensor, torch.Tensor]): The hidden state of the network. If None, the hidden state
                of the network is used. If provided, the hidden state of the neural network will not be updated. To
                retrieve the new hidden state, use the last_hidden_state property. If the network is not recurrent,
                this argument is ignored.
        Returns:
            The output of the network as a torch.Tensor.
        """
        assert hidden_state is None or self._recurrent, "Cannot pass hidden state to non-recurrent network."
        input = self._normalization(x.to(self.device))

        if self._recurrent:
            current_hidden_state = self.hidden_state if hidden_state is None else hidden_state
            current_hidden_state = (current_hidden_state[0].to(self.device), current_hidden_state[1].to(self.device))

            input = input.unsqueeze(0) if len(input.shape) == 2 else input
            input, next_hidden_state = self._features[0](input, current_hidden_state)
            input = self._features[1](input).squeeze(0)

            if hidden_state is None:
                self.hidden_state = next_hidden_state
            self._last_hidden_state = next_hidden_state
        
        # Forawrd passing shared pre-head layers
        shared_feature_pre_head = self.get_submodule(self._shared_pre_head_module_name)(input)
        # Forward passing each head layers
        batch_size = x.shape[0]
        head_num = len(self._separate_multi_head_module_name_list)
        multi_head_output = []
        for head_name in self._separate_multi_head_module_name_list:
            head_module = self.get_submodule(head_name)
            head_output = head_module(shared_feature_pre_head)
            multi_head_output.append(head_output)
        
        multi_head_output_tensor = squeeze_preserve_batch(torch.stack(multi_head_output, dim=1))
        output_tensor = multi_head_output_tensor.sum(axis=1)
        return output_tensor, multi_head_output_tensor

    @property
    def last_hidden_state(self):
        """Returns the hidden state of the last forward pass.

        Does not differentiate whether the hidden state depends on the hidden state kept in the network or whether it
        was passed into the forward pass.

        Returns:
            The hidden state of the last forward pass as Tuple[torch.Tensor, torch.Tensor].
        """
        return self._last_hidden_state

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the given input.

        Args:
            x (torch.Tensor): The input to normalize.
        Returns:
            The normalized input as a torch.Tensor.
        """
        output = self._normalization(x.to(self.device))

        return output

    @property
    def recurrent(self) -> bool:
        """Returns whether the network is recurrent."""
        return self._recurrent

    def reset_hidden_state(self, indices: torch.Tensor) -> None:
        """Resets the hidden state of the neural network.

        Throws an error if the network is not recurrent.

        Args:
            indices (torch.Tensor): A 1-dimensional int tensor containing the indices of the terminated
                environments.
        """
        assert self._recurrent

        self.hidden_state[0][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)
        self.hidden_state[1][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)

    def reset_full_hidden_state(self, batch_size=None) -> None:
        """Resets the hidden state of the neural network.

        Args:
            batch_size (int): The batch size of the hidden state. If None, the hidden state is reset to None.
        """
        assert self._recurrent

        if batch_size is None:
            self.hidden_state = None
        else:
            layer_count, hidden_size = self._features[0].num_layers, self._features[0].hidden_size
            self.hidden_state = (
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
            )

    def _init(self, shared_pre_head_module_name:str, separate_multi_head_module_name_list:List[str], fade: bool = True, gain: float = 1.0) -> None:
        """Initializes neural network layers."""
        
        # Initialize shared_pre_head_layers
        for idx, layer in enumerate(self.get_submodule(shared_pre_head_module_name)):
            if not isinstance(layer, nn.Linear):
                continue
            current_gain = gain    # Only the output layer is treated differently
            nn.init.xavier_normal_(layer.weight, gain=current_gain)
        
        # Initialize separate_multi_head_layers
        for head_name in separate_multi_head_module_name_list:
            head_layers = self.get_submodule(head_name)
            # Get the output layer indx (only consider layer with weights not activation layer)
            last_layer_idx = len(head_layers) - 1 - next(i for i, l in enumerate(reversed(head_layers)) if isinstance(l, nn.Linear))

            for h_idx, h_layer in enumerate(head_layers):
                if not isinstance(h_layer, nn.Linear):
                    continue

                current_gain = gain / 100.0 if fade and h_idx == last_layer_idx else gain    # Only the output layer is treated differently
                nn.init.xavier_normal_(h_layer.weight, gain=current_gain)
            
        return None
