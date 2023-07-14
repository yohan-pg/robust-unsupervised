from .degradations import *
import itertools

task_names = ["upsampling", "denoising", "deartifacting", "inpainting"]
task_levels = ["XL", "L", "M", "S", "XS"]

degradation_types = {
    "upsampling": Downsample,
    "inpainting": MaskRandomly,
    "denoising": AddNoise,
    "deartifacting": CompressJPEG,
}

degradation_levels = {
    "upsampling": {
        "XL": 32,
        "L": 16,
        "M": 8,
        "S": 4,
        "XS": 2,
        #
        2: 8,
        3: 8,
        4: 8,
    },
    "inpainting": {
        "XL": 17,
        "L": 13,
        "M": 9,
        "S": 5,
        "XS": 1,
        #
        2: 9,
        3: 9,
        4: 9,
    },
   "denoising": {
       "XL": (6,  0.64),
       "L":  (12, 0.32),
       "M":  (24, 0.16),
       "S":  (48, 0.08),
       "XS": (96, 0.04),
       #
       2: (24, 0.16),
       3: (24, 0.16),
       4: (24, 0.16),
    },
    "deartifacting": {
        "XL": 6,
        "L": 9,
        "M": 12,
        "S": 15,
        "XS": 18,
        #
        2: 12,
        3: 12,
        4: 12,
    },
}

####


@dataclass
class Task:
    name: str
    category: str
    level: str
    constructor: Type[Degradation]
    arg: Any

    def init_degradation(self):
        return ComposedDegradation(
            [ResizePrediction(config.resolution), self.constructor(self.arg)]
        )


def get_task(name: str, level: str):
    return Task(
        name,
        "single_tasks",
        level,
        degradation_types[name],
        degradation_levels[name][level],
    )


####

single_tasks: List[Task] = []
for level in task_levels:
    for name in task_names:
        single_tasks.append(get_task(name, level))

####


def init_composed(level: int):
    return lambda included_tasks: ComposedDegradation(
        [
            degradation_types[name](degradation_levels[name][level])
            for name in task_names
            if name in included_tasks
        ]
    )


full_composed_task = Task("UNAP", "composed_tasks", 4, init_composed(4), task_names)

####

initials = {
    "upsampling": "U",
    "denoising": "N",
    "deartifacting": "A",
    "inpainting": "P",
}

composed_tasks: List[Task] = []

for k in range(2, len(task_names) + 1):
    for task_names_subseq in itertools.combinations(task_names, k):
        composed_tasks.append(
            Task(
                "".join([initials[task_name] for task_name in task_names_subseq]),
                "composed_tasks",
                k,
                init_composed(k),
                task_names_subseq,
            )
        )

all_tasks: List[Task] = single_tasks + composed_tasks

extreme_tasks = []
extreme_tasks += [get_task(name, "XL") for name in task_names]
extreme_tasks += [full_composed_task]
extreme_tasks += [get_task(name, "XS") for name in task_names]


###

uncropping_task = Task("uncropping", "uncropping", 1, CenterCrop, None) 
identity_task = Task("identity", "identity", 1, IdentityDegradation, None)
