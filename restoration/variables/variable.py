from ..prelude import *


class Variable(nn.Module):
    init_truncation = 1.0
    random_noise = False

    def __init__(self, G, data: torch.Tensor):
        super().__init__()
        self.G = G
        self.data = data

    # ------------------------------------
    
    @staticmethod
    def sample_from(G, batch_size: int = 1):
        raise NotImplementedError

    @staticmethod
    def sample_random_from(G, batch_size: int = 1):
        raise NotImplementedError

    def to_input_tensor(self):
        raise NotImplementedError

    # ------------------------------------

    def parameters(self):
        return [self.data]

    def to_image(self):
        return self.render_image(self.to_input_tensor())

    def render_image(self, styles):
        return (
            self.G.synthesis(
                styles,
                noise_mode="random" if self.random_noise else "const",
                force_fp32=not config.fp16,
            )
            + 1.0
        ) / 2.0

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        return self.__class__(
            self.G,
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data,
        )

    def clone(self):
        data = self.detach().data.clone()
        return self.__class__(
            self.G,
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data,
        )

    def interpolate(self, other: "Variable", alpha=0.5):
        assert self.G == other.G
        return self.__class__(self.G, self.data.lerp(other.data, alpha))

    def clone(self):
        data = copy.deepcopy(self.data)
        return self.__class__(
            self.G,
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else data,
        )

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def unbind(self):
        return [
            self.__class__(
                self.G,
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]
