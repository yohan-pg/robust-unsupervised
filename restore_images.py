import restoration_config as config
from restoration.io_utils import open_generator
from restoration.restoration import run_experiment

run_experiment(open_generator(config.pkl, float=True, refresh=True))