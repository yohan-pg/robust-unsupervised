from ESIR.prelude import *
from ESIR.variables import *

import pickle
import shutil

import torch_utils.misc as misc
import contextlib


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


def open_target(G, *paths: str) -> None:
    images = []
    for path in paths:
        target_pil = PIL.Image.open(path).convert("RGB")
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(
            ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        )
        target_pil = target_pil.resize(
            (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
        )
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        images.append(
            torch.tensor(target_uint8.transpose([2, 0, 1])).cuda().unsqueeze(0) / 255
        )
    return torch.cat(images)


def open_image(path: str, resolution: Optional[int] = None) -> None:
    image = PIL.Image.open(path)
    if resolution is not None:
        image = image.resize((resolution, resolution), PIL.Image.LANCZOS)
    return TF.to_tensor(image).unsqueeze(0).cuda()[:, :3]


@torch.no_grad()
def sample_image(G, batch_size: int = 1):
    var = torch.randn(batch_size, G.mapping.z_dim)
    return (G.synthesis(var.to_styles()) + 1) / 2


def fresh_dir(path) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def resize_for_logging(x, resolution=config.resolution):
    return F.interpolate(
        x,
        size=(resolution, resolution),
        mode="nearest" if x.shape[-1] <= resolution else "area",
    )


@contextlib.contextmanager
def directory(dir_path: str, replace=False, must_not_exist=False):
    if replace:
        fresh_dir(dir_path)
    else:
        if must_not_exist:
            assert not os.path.exists(dir_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    cwd = os.getcwd()
    os.chdir(dir_path)
    yield
    os.chdir(cwd)
