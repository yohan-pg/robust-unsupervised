import benchmark
from benchmark import Task, Degradation

from ESIR.logging import *
from ESIR.variables import *
from ESIR.io_utils import *
from ESIR.dataloader import *
from ESIR.init import *

benchmark.config.resolution = config.resolution


def restore(G, loss_fn: Callable, task: Task):
    def invert(label: str, variable: Variable, lr: float, include_noises=False, adam=config.adam):
        if adam:
            optimizer = optim.Adam(list(variable.parameters()), lr=lr * config.adam_lr_scale)
        else: 
            optimizer = NGD(
                list(variable.parameters()) if include_noises else variable.parameters(), 
                lr=lr
            )

        pbar = tqdm.tqdm(
            range(config.num_steps * int(1.5 if config.no_Wp else 1)),
            desc=label,   
        )

        losses = []

        try:
            for j in pbar:
                x = variable.to_image()
                loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask, last_pass=label=="W++").mean()

                if config.warmup != 0:
                    for group in optimizer.param_groups:
                        group["lr"] = min(1, j / max(1, int(config.num_steps - 1 * config.warmup))) * lr

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                if config.trace:
                    if j % 10 == 0:
                        save_image(x, f"frame_{label}_{j:04d}.png")

                pbar.set_postfix(
                    {"loss": loss.item(), "lr": optimizer.param_groups[-1]["lr"]}
                )
        except KeyboardInterrupt:
            pass

        log(label, variable)

    def log(label, variable):
        logging(
            variable=variable,
            degradation=degradation,
            ground_truth=ground_truth,
            target=target,
            suffix="_" + label,
        )

    for j, ground_truth in enumerate(iterate_ground_truths()):
        with directory(f"inversions/{j:04d}", replace=True):
            print(f"- {j:04d}")

            degradation = task.init_degradation()
            save_image(ground_truth, f"ground_truth.png")
            target = degradation.degrade_ground_truth(ground_truth)
            save_image(target, f"target.png")
            
            W_variable = WVariable.sample_from(G)
            invert("W", W_variable, config.lr_W, adam=config.adam_phase=="W")

            Wp_variable = WpVariable.from_W(W_variable)
            if not config.no_Wp:
                invert("W+", Wp_variable, config.lr_Wp, adam=config.adam_phase=="W+") 

            Wpp_variable = WppVariable.from_Wp(Wp_variable)
            invert("W++", Wpp_variable, config.lr_Wpp, adam=config.adam_phase=="W++")

            combine_artifacts()


class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                assert param.isnan().sum().item() == 0
                g = param.grad
                g /= g.norm(dim=-1, keepdim=True)
                g = torch.nan_to_num(
                    g, nan=0.0, posinf=0.0, neginf=0.0
                )
                param -= group["lr"] * g


def run_experiment(G, label=""):
    from ESIR.loss_function import MultiscaleLPIPS

    print(config.name)

    loss_fn = MultiscaleLPIPS(
        level_weights=config.level_weights,
    )

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    experiment_name = f"{config.name}/{timestamp}/{label}"

    for task in config.tasks:
        path = f"out/{experiment_name}/{task.category}/{task.name}/{task.level}/"
        with directory(path):
            print(path)
            restore(G, loss_fn, task)
    
    return f"out/{experiment_name}"
