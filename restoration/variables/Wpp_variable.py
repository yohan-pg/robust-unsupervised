from .Wp_variable import *


class WppVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.sample_from(G, batch_size).to_input_tensor().repeat(1, 512, 1)

        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            WVariable.sample_random_from(G, batch_size)
            .to_input_tensor()
            .repeat(1, 512, 1)
        )

        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def from_w(W: WVariable):
        data = W.data.detach().repeat(1, 512 * W.G.num_ws, 1)

        return WppVariable(W.G, nn.parameter.Parameter(data))

    @staticmethod
    def from_Wp(Wp: WpVariable):
        data = Wp.data.detach().repeat_interleave(512, dim=1)

        return WppVariable(Wp.G, nn.parameter.Parameter(data))

    def to_input_tensor(self):
        return self.data

class WCombinedVariable(Variable):
    def parameters(self):
        return [self.data["w"], self.data["wp"], self.data["wpp"]]

    @staticmethod
    @torch.no_grad()
    def sample_from(G: nn.Module, batch_size: int = 1):
        w = WVariable.sample_from(G, batch_size)
        wp = WpVariable.from_W(w)
        wpp = WppVariable.from_Wp(wp)

        return WCombinedVariable(G, nn.ParameterDict({
            "w": nn.Parameter(w.data),
            "wp": nn.Parameter(wp.data * 0),
            "wpp": nn.Parameter(wpp.data * 0),
        }))

    def to_input_tensor(self):
        return (
            self.data["w"].repeat(1, self.data["wpp"].shape[1], 1) 
            + self.data["wp"].repeat(1, 512, 1) / 4.24
            + self.data["wpp"] / 74.6391
        )



class WStarVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.sample_from(G, batch_size).to_input_tensor().repeat(1, 512, 1)

        return WStarVariable(G, nn.Parameter(data * 0))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            WVariable.sample_random_from(G, batch_size)
            .to_input_tensor()
            .repeat(1, 512, 1)
        )

        return WStarVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        maps = [self.data]
        d = self.data.transpose(1, 2)

        for _ in range(9):
            d = F.avg_pool1d(d, 2, stride=2) * 2 * math.sqrt(2)
            maps.append(d.transpose(1, 2))
        
        fs = []
        for i, map in enumerate(maps):
            fs.append(map.repeat_interleave(2**i, dim=1))

        o = torch.stack(fs).mean(dim=0)

        return WVariable.sample_from(self.G, 1).data + o




# class WppVariableSpatial(Variable):
#     @staticmethod
#     def sample_from(G: nn.Module, batch_size: int = 1):
#         return WppVariableSpatial(
#             G,
#             nn.Parameter(
#                 WVariable.sample_from(G, batch_size)
#                 .to_input_tensor()
#                 .repeat(1, 512 * 9, 1)
#             ),
#         )

#     @staticmethod
#     def sample_random_from(G: nn.Module, batch_size: int = 1):
#         var = WppVariableSpatial(
#             G,
#             nn.Parameter(
#                 WVariable.sample_random_from(G, batch_size)
#                 .to_input_tensor()
#                 .repeat(1, 512 * 9, 1)
#             ),
#         )

#         return var

#     @staticmethod
#     def from_w(W: WVariable):
#         return WppVariable(
#             W.G,
#             nn.parameter.Parameter(W.data.detach().repeat(1, 9 * 512 * W.G.num_ws, 1)),
#         )

#     @staticmethod
#     def from_Wp(Wp: WpVariable):
#         return WppVariable(
#             Wp.G,
#             nn.parameter.Parameter(Wp.data.detach().repeat_interleave(512 * 9, dim=1)),
#         )

#     def to_input_tensor(self):
#         return self.data
