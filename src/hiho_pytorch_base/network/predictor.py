from torch import Tensor, nn

from hiho_pytorch_base.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def create_predictor(config: NetworkConfig) -> Predictor:
    return Predictor(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
    )