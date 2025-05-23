import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

CALLBACKS = {
}

def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks