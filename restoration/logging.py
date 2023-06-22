from ESIR.prelude import *
from ESIR.io_utils import *

import plotly.express as px


def plot_losses(label, losses, image_distances, degraded_distances, latent_distances):
    fig = px.line(losses)
    fig.update_layout(yaxis_range=[0, max(losses)])
    fig.write_image(f"losses_{label}.jpg")

    fig = px.line(image_distances)
    fig.update_layout(yaxis_range=[0, max(image_distances)])
    fig.write_image(f"image_distances_{label}.jpg")

    fig = px.line(degraded_distances)
    fig.update_layout(yaxis_range=[0, max(degraded_distances)])
    fig.write_image(f"degraded_distances_{label}.jpg")

    fig = px.line(latent_distances)
    fig.update_layout(yaxis_range=[0, max(latent_distances)])
    fig.write_image(f"latent_distances_{label}.jpg")

    ratios = [a / b for a, b in zip(image_distances, degraded_distances)]
    fig = px.line(ratios)
    fig.update_layout(yaxis_range=[0, max(ratios)])
    fig.write_image(f"ratios_{label}.jpg")


@torch.no_grad()
def logging(*, variable, suffix, degradation, ground_truth, target):
    pred = resize_for_logging(variable.to_image())

    approx_degraded_pred = degradation.degrade_prediction(pred)
    degraded_pred = degradation.degrade_ground_truth(pred)

    save_image(pred, f"pred{suffix}.png", padding=0)
    save_image(degraded_pred, f"degraded_pred{suffix}.png", padding=0)

    save_image(
        torch.cat([approx_degraded_pred, degraded_pred]),
        f"degradation_approximation{suffix}.jpg",
        padding=0,
    )

    save_image(
        torch.cat(
            [
                ground_truth,
                resize_for_logging(target),
                resize_for_logging(degraded_pred),
                pred,
            ]
        ),
        f"side_by_side{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([resize_for_logging(target), pred]),
        f"result{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([target, degraded_pred, (target - degraded_pred).abs()]),
        f"fidelity{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([ground_truth, pred, (ground_truth - pred).abs()]),
        f"accuracy{suffix}.jpg",
        padding=0,
    )


all_spaces = ["W", "W+", "W++"]


def combine_artifacts():
    for name, nrow, ext in [
        ("pred", 3, "png"),
        ("degraded_pred", 3, "png"),
        ("side_by_side", 1, "jpg"),
        ("result", 1, "jpg"),
        ("accuracy", 1, "jpg"),
        ("fidelity", 1, "jpg"),
    ]:
        save_image(
            torch.cat([open_image(f"{name}_{suffix}.{ext}") for suffix in all_spaces]),
            f"combined_{name}.jpg",
            nrow=nrow,
            padding=0,
        )


def render_mosaics():
    num_images = min(16, config.max_images or sys.maxsize)

    for suffix in all_spaces:
        save_image(
            torch.cat(
                [
                    open_image(f"inversions/{j:04d}/pred_{suffix}.png")
                    for j in range(num_images)
                ]
            ),
            f"pred_{suffix}.jpg",
            nrow=4,
            padding=0,
        )
        num_images = min(num_images, 6)
        for name in ["result", "fidelity", "accuracy", "side_by_side"]:
            save_image(
                torch.cat(
                    [
                        open_image(f"inversions/{j:04d}/{name}_{suffix}.jpg")
                        for j in range(num_images)
                    ]
                ),
                f"{name}_{suffix}.jpg",
                nrow=1,
                padding=0,
            )
