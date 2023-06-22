from .prelude import *
import lpips


class MultiscaleLPIPS:
    _lpips = None

    def __init__(
        self,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__()
        self.weights = level_weights
        if MultiscaleLPIPS._lpips is None:
            MultiscaleLPIPS._lpips = lpips.LPIPS(net="vgg", verbose=False).cuda()

    def lpips(self, x, y, mask):
        if mask is not None:
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)

        return self._lpips(x, y, normalize=True).mean() 

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)

        losses = []

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        for weight in self.weights:
            if y.shape[-1] <= config.min_loss_res:
                break

            if weight > 0:
                loss = self.lpips(x, y, mask)
                losses.append(weight * loss)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)
        
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = config.l1_weight * F.l1_loss(x, y)

        return total + l1
