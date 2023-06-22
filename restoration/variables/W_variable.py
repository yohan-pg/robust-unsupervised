from .variable import *


class WVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping.w_avg.reshape(1, G.w_dim).repeat(batch_size, 1)

        # if config.one_init:
        #     data = torch.ones_like(data) 

        return WVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping(
            torch.randn(batch_size, G.z_dim).cuda(),
            None,
            skip_w_avg_update=True,
        )[:, 0]

        return WVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data.unsqueeze(1).repeat(1, self.G.num_ws, 1)

    @torch.no_grad()
    def truncate(self, truncation=1.0):
        assert 0.0 <= truncation <= 1.0
        self.data.lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
        return self



class VariableDiffusion(nn.Module):
    init_truncation = 1.0
    random_noise = False

    def __init__(self, VAE, data: torch.Tensor):
        super().__init__()
        self.VAE = VAE
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
        return self.VAE.decode(styles).sample / 2 + 0.5

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        return self.__class__(
            self.VAE,
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data,
        )

    def clone(self):
        data = self.detach().data.clone()
        return self.__class__(
            self.VAE,
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data,
        )

    def interpolate(self, other: "Variable", alpha=0.5):
        assert self.VAE == other.G
        return self.__class__(self.VAE, self.data.lerp(other.data, alpha))

    def clone(self):
        data = copy.deepcopy(self.data)
        return self.__class__(
            self.VAE,
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
                self.VAE,
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]


from PIL import Image

class WVariableDiffusion(VariableDiffusion):
    @staticmethod
    @torch.no_grad()
    def sample_from(VAE: nn.Module, batch_size: int = 1):
        x = F.interpolate(TF.to_tensor(Image.open("/home-local2/yopog.extra.nobkp/paper/esir/datasets/mean_img.png"))[None].cuda(), 512) * 2 - 1
        VAE.mean = VAE.encode(x).latent_dist.mean
        return WVariableDiffusion(VAE, nn.parameter.Parameter(torch.ones(512)))
    
    def parameters(self):
        params = []
        for module in self.VAE.decoder.mid_block.modules():
            if isinstance(module, nn.Conv2d):
                params.append(module.parameters())
        return itertools.chain(*params)

    def to_input_tensor(self):
        return self.VAE.mean# + F.interpolate(self.data, self.VAE.mean.shape[-1])


# class W2Variable(Variable):
#     split_point = 5

#     @staticmethod
#     def sample_from(G: nn.Module, batch_size: int = 1):
#         data = WVariable.sample_from(G, batch_size).data.unsqueeze(1).repeat(1, 2, 1)

#         return W2Variable(G, nn.Parameter(data))

#     @staticmethod
#     def sample_random_from(G: nn.Module, batch_size: int = 1):
#         data = (
#             WVariable.sample_random_from(G, batch_size)
#             .data.unsqueeze(1)
#             .repeat(1, 2, 1)
#         )
#         return W2Variable(G, nn.Parameter(data))

#     def to_input_tensor(self):
#         return torch.cat([
#             self.data[:, 0:1].repeat(1, self.split_point, 1),
#             self.data[:, 1:2].repeat(1, self.G.num_ws - self.split_point, 1)
#         ], dim=1)

#     @torch.no_grad()
#     def truncate(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data.lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self

#     @torch.no_grad()
#     def truncate_coarse(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data[:, 0:1].lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self

#     @torch.no_grad()
#     def truncate_fine(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data[:, 1:2].lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self



# class W3Variable(Variable):
#     split_point_coarse = 3
#     split_point_fine = 8

#     @staticmethod
#     def sample_from(G: nn.Module, batch_size: int = 1):
#         data = WVariable.sample_from(G, batch_size).data.unsqueeze(1).repeat(1, 3, 1)

#         return W3Variable(G, nn.Parameter(data))

#     @staticmethod
#     def sample_random_from(G: nn.Module, batch_size: int = 1):
#         data = (
#             WVariable.sample_random_from(G, batch_size)
#             .data.unsqueeze(1)
#             .repeat(1, 3, 1)
#         )
#         return W3Variable(G, nn.Parameter(data))

#     def to_input_tensor(self):
#         return torch.cat([
#             self.data[:, 0:1].repeat(1, self.split_point_coarse, 1),
#             self.data[:, 1:2].repeat(1, self.split_point_fine - self.split_point_coarse, 1),
#             self.data[:, 2:3].repeat(1, self.G.num_ws - self.split_point_fine, 1)
#         ], dim=1)

#     @torch.no_grad()
#     def truncate(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data.lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self

#     @torch.no_grad()
#     def truncate_coarse(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data[:, 0:1].lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self

#     @torch.no_grad()
#     def truncate_medium(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data[:, 1:2].lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self

#     @torch.no_grad()
#     def truncate_fine(self, truncation=1.0):
#         assert 0.0 <= truncation <= 1.0
#         self.data[:, 2:3].lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
#         return self
