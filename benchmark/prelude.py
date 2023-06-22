import copy
import os

import functools
from functools import partial
from time import perf_counter
import sys
import torch.optim as optim
import tqdm
import click
import dataclasses
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from xml.dom import minidom
import shutil
import itertools
from warnings import warn

from torchvision.utils import save_image, make_grid

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass, field

from typing import Optional, Type, List, final, Tuple, Callable, Iterator, Iterable, Dict, ClassVar, Union, Any

from torchvision.io import write_video
