import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def save_output_dtype(name: str, observed_dtypes: dict[str, torch.dtype]):
    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed_dtypes[name] = output.dtype

    return hook


def main() -> None:
    device = "cuda"
    model = ToyModel(in_features=16, out_features=64).to(device)
    x = torch.randn(4, 16, dtype=torch.float32, device=device)
    targets = torch.randint(0, 64, (4,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    observed_dtypes: dict[str, torch.dtype] = {}
    model.fc1.register_forward_hook(save_output_dtype("fc1_output", observed_dtypes))
    model.ln.register_forward_hook(save_output_dtype("layer_norm_output", observed_dtypes))

    with torch.autocast(device_type=device, dtype=torch.float16):
        parameter_dtype = next(model.parameters()).dtype
        logits = model(x)
        loss = loss_fn(logits, targets)

    loss.backward()

    print(f"model parameters in autocast context: {parameter_dtype}")
    print(f"output of fc1: {observed_dtypes['fc1_output']}")
    print(f"output of LayerNorm: {observed_dtypes['layer_norm_output']}")
    print(f"predicted logits: {logits.dtype}")
    print(f"loss: {loss.dtype}")
    print(f"gradients: {model.fc1.weight.grad.dtype}")


if __name__ == "__main__":
    main()
