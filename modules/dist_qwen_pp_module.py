from torch import nn
# from .qwen_modules import Qwen2Embeddings, Qwen2DecoderLayer, Qwen2LMHead
from .qwen_modules import Qwen2LMHead, Qwen2Embeddings, Qwen2DecoderLayer
# from transformers.models.qwen2.modeling_qwen2 import 


class Qwen2StageBase(nn.Module):
    def __init__(self, args, config):
        super(Qwen2StageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self._embedding_dim = args.embedding_dim
        self._seq_length = args.seq_length
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads
        self._num_layers = args.num_layers
        self._task_type = getattr(args, 'task_type', 'language_model')
        self.config = config

    def _create_first_layer(self):
        return Qwen2Embeddings(self.config)

    def _create_last_layer(self):
        if self._task_type == 'language_model':
            return Qwen2LMHead(self.config)
        raise Exception('Only language_model task type is supported for Qwen2')

    def _create_transformer_layer(self, layer_idx):
        # return Qwen2DecoderLayer(self.config, layer_idx=layer_idx, use_checkpoint=True)
        return Qwen2DecoderLayer(self.config, layer_idx=layer_idx)


class Qwen2StageFirst(Qwen2StageBase):
    def __init__(self, args, config, device):
        super(Qwen2StageFirst, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for i in range(self._num_layers):
            module_list.append(self._create_transformer_layer(i))
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class Qwen2StageMiddle(Qwen2StageBase):
    def __init__(self, args, config, device):
        super(Qwen2StageMiddle, self).__init__(args, config)
        self.device = device
        module_list = []
        for i in range(self._num_layers):
            module_list.append(self._create_transformer_layer(i + args.num_layers * args.pipeline_group_size))
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class Qwen2StageLast(Qwen2StageBase):
    def __init__(self, args, config, device):
        super(Qwen2StageLast, self).__init__(args, config)
        self.device = device
        module_list = []
        start_layer_idx = args.num_layers * (args.pipeline_group_size - 1)
        for i in range(self._num_layers):
            module_list.append(self._create_transformer_layer(start_layer_idx + i))
        module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out