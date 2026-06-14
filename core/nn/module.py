import numpy as np
from core.autodiff.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, Tensor): data = data.data
        if not isinstance(data, np.ndarray): data = np.array(data, dtype=np.float32)
        super().__init__(data, requires_grad=True)

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = list(self._parameters.values())
        for m in self._modules.values(): params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        items = []
        for name, param in self._parameters.items():
            items.append((f"{prefix}.{name}" if prefix else name, param))
        for name, module in self._modules.items():
            items.extend(module.named_parameters(f"{prefix}.{name}" if prefix else name))
        return items

    def train(self):
        self.training = True
        for m in self._modules.values(): m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values(): m.eval()
        return self

    def __setattr__(self, name, value):
        if isinstance(value, Parameter): self._parameters[name] = value
        elif isinstance(value, Module): self._modules[name] = value
        super().__setattr__(name, value)

    def state_dict(self):
        state = {}
        for name, param in self._parameters.items(): state[name] = param.data.copy()
        for name, module in self._modules.items():
            for pname, p in module.named_parameters(name): state[pname] = p.data.copy()
        return state

    def load_state_dict(self, state):
        for name, param in self._parameters.items():
            if name in state: param.data = state[name].copy()
        for name, module in self._modules.items():
            module.load_state_dict({k: v for k, v in state.items() if k.startswith(name + ".")})

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, param in self._parameters.items(): lines.append(f"  ({name}): Parameter({param.shape})")
        for name, module in self._modules.items(): lines.append(f"  ({name}): {module.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)
