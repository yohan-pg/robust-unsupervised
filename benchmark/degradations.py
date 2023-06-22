import benchmark.config as config

from .prelude import *

sys.path.append(os.path.dirname(__file__) + "/" + "DiffJPEG")
from DiffJPEG import DiffJPEG

import torchvision.transforms.functional as TF
from PIL import JpegImagePlugin
import tempfile
import random
import cv2
import os


TMP_SAVE_FILEPATH = tempfile.mkstemp()[1]


class Degradation(nn.Module):
    seed = 2022
    mask = None

    def __init__(self):
        super().__init__()
        self.seed += 1

    def _true_degradation(self, ground_truth):
        """
        Applies the true degradation, which may be non-differentiable.
        """
        raise NotImplementedError

    def degrade_prediction(self, pred):
        """
        Applies the differentiable approximation to the given degradation.
        """
        raise NotImplementedError

    @torch.no_grad()
    def degrade_ground_truth(self, ground_truth, save_path=None):
        """
        Applies the true, potentially undifferentiable degradation to a ground truth image.
        As a sanity check, the image is always saved to a file.
        """
        torch.manual_seed(self.seed)
        if save_path is None:
            save_path = TMP_SAVE_FILEPATH + ".png"

        degraded_target = self._true_degradation(ground_truth.clamp(min=0, max=1))
        result = cycle_to_file(degraded_target, save_path)
        torch.seed()
        return result

    def forward(self, x):
        return self.degrade_prediction(x)


def cycle_to_file(x, save_path: str):
    """
    Saves an image to a file and reads it back immediately.
    This is used to propely account for quantization and clamping,
    ensuring that the differentiable approx. does not make false assumptions.
    (For instance, clamping must be taken account for when adding strong noise)
    """
    assert x.shape[0] == 1  # batching is not supported yet
    TF.to_pil_image(x.squeeze(0).clamp(0, 1)).save(save_path)
    return TF.to_tensor(PIL.Image.open(save_path)).unsqueeze(0).to(x.device)


class Downsample(Degradation):
    def __init__(self, downsampling_factor: int):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.filter = random.choice(
            [
                PIL.Image.BILINEAR,
                PIL.Image.BICUBIC,
                PIL.Image.LANCZOS,
            ]
        )

    def degrade_prediction(self, x):
        return F.avg_pool2d(x, self.downsampling_factor)

    def _true_degradation(self, x):
        assert x.shape[0] == 1, "Batching not yet supported"
        image = TF.to_pil_image(x.squeeze(0))
        res = math.floor(x.shape[-1] // self.downsampling_factor)

        image = image.resize(
            (res, res),
            self.filter,
        )
        path = TMP_SAVE_FILEPATH + ".png"
        image.save(path)
        return TF.to_tensor(PIL.Image.open(path)).unsqueeze(0).to(x.device)


class AddNoise(Degradation):
    k = 2.0
    eps = 1e-3

    def __init__(
        self,
        noise_amount: float,
    ):
        super().__init__()
        self.noise_amount = noise_amount
        self.clamp = True
        self.seed += 1

    def degrade_prediction(self, x):
        # pre-clamp the generator output, to match the fact that the (artificial) noise is
        # added to ground truth images with pixel values in [0, 1]
        # if working with real sources of noise this can be omitted
        x = self.differentiable_clamp(x)

        num_photons, bernoulli_p = self.noise_amount

        # Approximate poisson noise with a gaussian
        if num_photons > 0:
            noise = torch.randn(1, 3, x.shape[2], x.shape[3], device=x.device)
            lambd = x * num_photons
            mu = lambd - 1 / 2
            sigma = (lambd + self.eps).sqrt()
            y = (mu + sigma * noise) / num_photons
        else:
            y = x

        # Bernoulli noise
        y = y * (torch.rand_like(y)[:, 0:1] > bernoulli_p).float()

        return self.differentiable_clamp(y)

    @torch.no_grad()
    def _true_degradation(self, x):
        num_photons, bernoulli_p = self.noise_amount

        # Add poisson noise
        if num_photons > 0:
            y = torch.poisson(x * num_photons) / num_photons
        else:
            y = x

        # Bernoulli noise
        y = y * (torch.rand_like(y)[:, 0:1] > bernoulli_p).float()

        return y.clamp(0.0, 1.0)

    class _ClampWithSurrogateGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(0.0, 1.0)

        @staticmethod
        def backward(ctx, grad_y):
            (x,) = ctx.saved_tensors
            with torch.enable_grad():
                return (
                    torch.autograd.grad(
                        torch.sigmoid(AddNoise.k * (x - 0.5)), x, grad_y
                    )[0],
                    None,
                )

    differentiable_clamp = _ClampWithSurrogateGradient.apply


class CenterCrop(Degradation):
    def __init__(self, *args):
        super().__init__()
        
    def degrade_prediction(self, x):
        result = torch.zeros_like(x)
        result[:, :, 400:600, 400:600] = x[:, :, 400:600, 400:600]
        return result

    def _true_degradation(self, x):
        return self.degrade_prediction(x)


class CompressJPEG(Degradation):
    k = 0.8

    def __init__(self, quality: int):
        super().__init__()
        self.quality = quality

        # Possible subsampling values are 0, 1 and 2 that correspond to 4:4:4, 4:2:2 and 4:2:0.
        x_img = TF.to_pil_image(torch.randn(3, config.resolution, config.resolution))

        # Extract quantization table
        path = TMP_SAVE_FILEPATH + ".jpg"
        x_img.save(
            path,
            quality=self.quality,
        )
        compressed_image = PIL.Image.open(path)
        table = compressed_image.quantization  # type: ignore
        assert JpegImagePlugin.get_sampling(compressed_image) == 2

        self.to_jpeg = DiffJPEG(
            self.k,
            differentiable=True,
            quantization_table=table,
        ).cuda()  # type: ignore

    def parameters(self, recurse=False):
        # This is important, it prevents the optimization of DiffJPEG's parameters
        return []

    def degrade_prediction(self, x):
        return self.to_jpeg(x)

    def _true_degradation(self, x):
        if "CHEAT_DEARTIFACT" in os.environ:
            return self.degrade_prediction(x).detach()
        else:
            assert x.shape[0] == 1, "Batching not yet supported"
            path = TMP_SAVE_FILEPATH + ".jpg"
            TF.to_pil_image(x.squeeze(0)).save(path, quality=self.quality)
            return TF.to_tensor(PIL.Image.open(path)).unsqueeze(0).to(x.device)


class MaskRandomly(Degradation):
    def __init__(self, num_strokes: int):
        super().__init__()
        self.num_strokes = num_strokes
        torch.manual_seed(self.seed)
        self.mask = self._generate_mask()
        torch.seed()

    def _generate_mask(self):
        image_height = config.resolution * 4
        image_width = config.resolution * 4
        brush_width = int(config.resolution * 0.08) * 4

        mask = np.zeros((image_height, image_width))

        def sample():
            w = image_width - 1
            h = image_height - 1
            return random.choice(
                [
                    random.randint(0, w // 3),
                    random.randint(2 * w // 3, w),
                ]
            ), random.choice(
                [
                    random.randint(0, h // 3),
                    random.randint(2 * h // 3, h),
                ]
            )

        for _ in range(self.num_strokes):
            start_x, start_y = sample()
            end_x, end_y = sample()
            mask = cv2.line(
                mask,
                (start_x, start_y),
                (end_x, end_y),
                color=1,
                thickness=brush_width,
            )
            mask = cv2.circle(mask, (start_x, start_y), int(brush_width / 2), 1)

        return (
            torch.from_numpy(1.0 - cv2.pyrDown(cv2.pyrDown(mask)))
            .float()
            .cuda()[None, None]
        )

    def _true_degradation(self, x):
        return x * F.interpolate(
            self.mask, x.shape[-1], mode="bicubic", align_corners=False
        )

    def degrade_prediction(self, x):
        return self._true_degradation(x)


class IdentityDegradation(Degradation):
    def __init__(self, *args):
        super().__init__()

    def degrade_prediction(self, x):
        return x

    def _true_degradation(self, x):
        return x


class ComposedDegradation(Degradation):
    def __init__(
        self,
        degradations: List[Degradation],
    ):
        super().__init__()
        self.degradations = nn.ModuleList(degradations)

    @property
    def mask(self):
        return self.degradations[-1].mask

    def parameters(self, recurse=False):
        return sum([list(deg.parameters()) for deg in self.degradations], [])

    def degrade_prediction(self, x):
        for deg in self.degradations:
            x = deg.degrade_prediction(x)
        return x

    def _true_degradation(self, x):
        for deg in self.degradations:
            x = deg._true_degradation(x)
        return x

    def degrade_ground_truth(self, x, save_path=None):
        for deg in self.degradations:
            # Re-saves the image between each degradation, overkill but OK
            x = deg.degrade_ground_truth(x, save_path=save_path)
        return x


class ResizePrediction(Degradation):
    # This is a hack used to match the prediction resolution to the target resolutions

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def degrade_prediction(self, x):
        return self._true_degradation(x)

    def _true_degradation(self, x):
        return F.interpolate(
            x,
            size=self.size,
            mode="area",
        )


def adapt_to_resolution(x, res: int):
    return ComposedDegradation([ResizePrediction(res), x])
