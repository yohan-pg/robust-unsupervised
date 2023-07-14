from robust_unsupervised.prelude import *
from robust_unsupervised.variables import *

import shutil
import torch_utils as torch_utils
import torch_utils.misc as misc
import contextlib

import PIL.Image as Image


def open_generator(pkl_path: str, refresh=True, float=True, ema=True) -> networks.Generator:
    print(f"Loading generator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        G = legacy.load_network_pkl(fp)["G_ema" if ema else "G"].cuda().eval()
        if float:
            G = G.float()

    if refresh:
        with torch.no_grad():
            old_G = G
            G = networks.Generator(*old_G.init_args, **old_G.init_kwargs).cuda()
            misc.copy_params_and_buffers(old_G, G, require_all=True)
            for param in G.parameters():
                param.requires_grad = False

    return G


def open_image(path: str, resolution: int):
    image = TF.to_tensor(Image.open(path)).cuda().unsqueeze(0)[:, :3]
    image = TF.center_crop(image, min(image.shape[2:]))
    return F.interpolate(image, resolution, mode="area")


def resize_for_logging(x: torch.Tensor, resolution: int) -> torch.Tensor:
    return F.interpolate(
        x,
        size=(resolution, resolution),
        mode="nearest" if x.shape[-1] <= resolution else "area",
    )


@contextlib.contextmanager
def directory(dir_path: str) -> None: 
    "Context manager for entering a directory, while automatically creating it if it does not exist."
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cwd = os.getcwd()
    os.chdir(dir_path)
    yield
    os.chdir(cwd)
