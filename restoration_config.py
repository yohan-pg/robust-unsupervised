import benchmark
import os

name = f"restored_samples"
pkl = "pretrained/ffhq.pkl"
dataset_path = "datasets/samples"
resolution = 1024
max_images = None

# -------------------
# By default all resotration tasks will run.

tasks = benchmark.all_tasks()

# Uncomment and modify to pick your own.
# tasks = [
#     benchmark.get_task("inpainting", "XS"),
#     benchmark.get_task("denoising", "S"),
#     benchmark.get_task("deartifacting", "M"),
#     benchmark.get_task("upsampling", "L"),
# ]

# Uncomment to run all composed tasks.
# tasks = benchmark.composed_tasks

# -------------------
# These hyperparameters should not require adjustment.

lr_W = 0.08 
lr_Wp = 0.02
lr_Wpp = 0.005

num_steps = 150
level_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
l1_weight = 0.1

min_loss_res = 16 

# -------------------
# This tells the benchmarking script what image resolution we are working with.

benchmark.config.resolution = resolution
