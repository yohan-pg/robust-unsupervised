import torch 


class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                assert param.isnan().sum().item() == 0
                g = param.grad
                g /= g.norm(dim=-1, keepdim=True)
                g = torch.nan_to_num(
                    g, nan=0.0, posinf=0.0, neginf=0.0
                )
                param -= group["lr"] * g