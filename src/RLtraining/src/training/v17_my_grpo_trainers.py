""" Custom GRPO Trainers. Extend from GRPOTrainer(Trainer), based on trl==0.17.0.dev0 """

from trl.trainer.grpo_trainer import GRPOTrainer

from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.import_utils import is_deepspeed_available, is_rich_available, is_vllm_available
from trl.trainer.grpo_trainer import nanstd
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)

#from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
#from transformers.utils import is_peft_available, is_torch_xla_available, is_sagemaker_mp_enabled

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, broadcast
from typing import Union, Any
from torch import nn
import torch
import warnings
import random
import wandb
import math

import transformers

from peft import PeftModel
import os

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

from packaging import version


def print_prompt_completions_sample(
    prompts: list[str], completions: list[str], rewards: dict[str, list[float]], step: int, num_samples: int = None
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
            Number of random samples to display. If `None` (default), all items will be displayed.

    """
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    #table.add_column("gold_diagnosis", style="bright_yellow")
    table.add_column("completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(completions):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(completions)), num_samples)
        #prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}

    for i in range(len(completions)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]  # 2 decimals
        #prompt_text = str(prompts[i]) if prompts[i] is not None else ""
        completion_text = str(completions[i]) if completions[i] is not None else ""
        #table.add_row(Text(prompt_text), Text(completion_text), *reward_values)
        table.add_row(Text(completion_text), *reward_values)
        #table.add_row(Text(prompts[i]), Text(completions[i]), *reward_values)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class MyCustomv17GRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer. Extend from GRPOTrainer(Trainer), based on trl==0.17
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super(GRPOTrainer, self).log(logs, start_time)
        else:  # transformers<=4.46
            super(GRPOTrainer, self).log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                #wandb.log({"completions": wandb.Table(dataframe=df)})