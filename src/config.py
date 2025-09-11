
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lookback: int = 28
    predict: int = 7
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 2
    emb_dim: int = 8
    dropout_p: float = 0.3
    folds: int = 10
    device: str = "cuda"
    zero_weight: float = 0.001  # for ScaledSMAPELoss

    def resolve_device(self):
        import torch
        return (self.device if self.device in ["cpu", "cuda"]
                else ("cuda" if torch.cuda.is_available() else "cpu"))
