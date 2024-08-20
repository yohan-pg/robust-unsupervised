import os
from pathlib import Path
from torchvision.io import read_image, write_png
import torchvision.transforms as T
import glob
from typing import * # type: ignore
import shutil
import subprocess
import json
from joblib import Parallel, delayed
import torchmetrics
import torch.nn.functional as F
import torch
import sys

DATASET_NAME = "FFHQ-X"
DATASET_PATH = "datasets"

CROP_RES = 128
CROP_RES_LABEL = "" if CROP_RES == 256 else str(CROP_RES)  # legacy

CROP_NUM = 1000 
CROP_NUM_LABEL = '_ncrops' + str(CROP_NUM) if CROP_NUM != 250 else ''

def globr(pattern):
    paths = glob.glob(pattern, recursive=True)
    assert len(paths) > 0, f"{pattern} matches nothing"
    return paths


def save_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_png(x, path)


def make_crops(
    image_paths: List[str],
    out_path: str,
    num_crops_per_image: int,
):
    cropping = T.RandomCrop(CROP_RES)
    print("Producing crops...")

    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)

    os.makedirs(out_path)

    @delayed
    def process_image(i, im_path):
        pred = read_image(im_path)
        for k in range(num_crops_per_image):
            save_image(cropping(pred), f"{out_path}/{i:04d}/{k:04}.png")

    Parallel(n_jobs=16, verbose=10)(process_image(*x) for x in enumerate(image_paths))


def crop_dataset():
    if True:
        make_crops(
            globr(f"{DATASET_PATH}/{DATASET_NAME}/**/*.png"),
            f"datasets/{DATASET_NAME}_crops{CROP_RES}{CROP_NUM_LABEL}",
            10 if "DRY_RUN" in os.environ else CROP_NUM,
        )
    if True:
        print("Evaluating FID...")
        os.system(
            f"python -m pytorch_fid --batch-size 50 --save-stats datasets/{DATASET_NAME}_crops{CROP_RES} benchmark/FFHQ-X_crops128_ncrops1000.npz"
        )


import torchmetrics
import torchmetrics.image.lpip as lpips


def accronym(metric):
    return "".join(x for x in metric.__class__.__name__ if not x.islower())


def replace(str, from_part, to_part):
    assert from_part in str
    return str.replace(from_part, to_part)


def eval_experiment(
    expr_path: str,
    suffixes: List[str],
    distance_metrics=[
        torchmetrics.PeakSignalNoiseRatio(data_range=2.0).cuda(),
        lpips.LearnedPerceptualImagePatchSimilarity(net_type="vgg").cuda(),
    ],
):
    for suffix in suffixes:
        if COMPUTE_FID := True:
            make_crops(
                globr(f"{expr_path}/**/pred{suffix}.png"),
                f"{expr_path}/crops{CROP_RES_LABEL}{suffix}",
                10 if "DRY_RUN" in os.environ else CROP_NUM,
            )
            result = subprocess.check_output(
                f"python -m pytorch_fid benchmark/FFHQ-X_crops128_ncrops1000.npz {expr_path}/crops{CROP_RES_LABEL}{suffix}".split(" ")
            )
            fid_score = float(result.decode("utf8").strip().replace("FID:  ", ""))
            json.dump(
                fid_score,
                open(
                    f"{expr_path}/fid{suffix.replace('/', '_')}{'.dry_run' if 'DRY_RUN' in os.environ else ''}{CROP_RES}{CROP_NUM_LABEL}.json",
                    "w",
                ),
            )

        if COMPUTE_DISTANCE_METRICS := False:
            degraded_scores = {accronym(metric): [] for metric in distance_metrics}
            ground_truth_scores = {accronym(metric): [] for metric in distance_metrics}

            for im_path in globr(f"{expr_path}/inversions/**/pred{suffix}.png"):
                def imopen(x):
                    return (read_image(x).unsqueeze(0).float() / 255.0) * 2.0 - 1.0

                pred = imopen(im_path)
                degraded_pred = imopen(replace(im_path, f"pred{suffix}", f"degraded_pred{suffix}"))
                target = imopen(replace(im_path, f"pred{suffix}", "target"))
                ground_truth = imopen(replace(im_path, f"pred{suffix}", "ground_truth"))

                for metric in distance_metrics:
                    degraded_scores[accronym(metric)].append(
                        metric(degraded_pred.cuda(), target.cuda()).item()
                    )
                    ground_truth_scores[accronym(metric)].append(
                        metric(pred.cuda(), ground_truth.cuda()).item()
                    )

                if "DRY_RUN" in os.environ:
                    break

            json.dump(
                {
                    name: torch.tensor(scores).mean().item()
                    for name, scores in degraded_scores.items()
                },
                open(
                    f"{expr_path}/degraded_scores{suffix.replace('/', '_')}{'.dry_run' if 'DRY_RUN' in os.environ else ''}.json",
                    "w",
                ),
            )
            json.dump(
                gtscores := {
                    name: torch.tensor(scores).mean().item()
                    for name, scores in ground_truth_scores.items()
                },
                open(
                    f"{expr_path}/ground_truth_scores{suffix.replace('/', '_')}{'.dry_run' if 'DRY_RUN' in os.environ else ''}.json",
                    "w",
                ),
            )
            print(gtscores["LPIPS"])


def eval_all_experiments(
    path: str,
    suffixes: List[str],
):
    for path in globr(f"{path}/**/inversions/"):
        print("👉", path)
        expr_path = path.split("/inversions")[0]
        eval_experiment(expr_path, suffixes)


if __name__ == "__main__":
    import sys 
    breakpoint()
    eval_all_experiments(sys.argv[1] + "/*", ["_W++"])