from restoration.prelude import *
from restoration.variables import *

import glob

import PIL.Image as Image

paths = sorted(
    [
        os.path.abspath(path)
        for path in (
            glob.glob(restoration_config.dataset_path + "/**/*.png", recursive=True)
            + glob.glob(restoration_config.dataset_path + "/**/*.jpg", recursive=True)
            + glob.glob(restoration_config.dataset_path + "/**/*.jpeg", recursive=True)
            + glob.glob(restoration_config.dataset_path + "/**/*.tif", recursive=True)
        )
    ]
)
assert len(paths) >= (restoration_config.max_images or 0)

def iterate_ground_truths():
    for path in paths[: restoration_config.max_images]:
        image = TF.to_tensor(Image.open(path)).cuda().unsqueeze(0)[:, :3]
        image = TF.center_crop(image, min(image.shape[2:]))
        yield F.interpolate(
            image,
            restoration_config.resolution or image.shape[-1],
            mode="area"
        )