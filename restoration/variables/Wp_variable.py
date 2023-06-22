from .W_variable import *


class WpVariable(Variable):
    def __init__(self, G, data: torch.Tensor):
        super().__init__(G, data)

    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.to_input_tensor(WVariable.sample_from(G, batch_size))

        return WpVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            G.mapping(
                (torch.randn(batch_size * G.mapping.num_ws, G.z_dim).cuda()),
                None,
                skip_w_avg_update=True,
            )
            .mean(dim=1)
            .reshape(batch_size, G.mapping.num_ws, G.w_dim)
        )

        return WpVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data

    def mix(self, other: "WpVariable", num_layers: float):
        return WpVariable(
            self.G,
            torch.cat(
                (self.data[:, :num_layers, :], other.data[:, num_layers:, :]), dim=1
            ),
        )

    @staticmethod
    def from_W(W: WVariable):
        return WpVariable(
            W.G, nn.parameter.Parameter(W.to_input_tensor())
        )

    @torch.no_grad()
    def truncate(self, truncation=1.0, *, layer_start = 0, layer_end: Optional[int] = None):
        assert 0.0 <= truncation <= 1.0
        mu = self.G.mapping.w_avg
        target = mu.reshape(1, 1, 512).repeat(1, self.G.mapping.num_ws, 1)
        self.data[:, layer_start:layer_end].lerp_(
            target[:, layer_start:layer_end], 1.0 - truncation
        )
        return self
