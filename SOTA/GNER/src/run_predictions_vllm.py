import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from gner_trainer import GNERTrainer
from gner_collator import DataCollatorForGNER
from gner_evaluator import compute_metrics

from src.SFT_finetuning.commons.initialization import init_model, wrap_model_for_peft, get_HF_access_token

# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams
from datasets import Dataset

# off wandb
os.environ['WANDB_DISABLED'] = "True"
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train/dev/test splits and labels."}
    )
    no_load_gner_customized_datasets: bool = field(
        default=False, metadata={"help": "Whether to load GNER datasets. If False, you should provide json files"}
    )
    train_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train data."}
    )
    valid_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the valid data."}
    )
    test_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the test data."}
    )
    data_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    from huggingface_hub import login
    HF_ACCESS_TOKEN = get_HF_access_token('./.env')
    login(token=HF_ACCESS_TOKEN)

    vllm_model = LLM(model=model_args.model_name_or_path, download_dir='./hf_cache_dir')
    max_new_tokens = 640  # as they require
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens) #, stop=['</s>'])

    test_datasets_names = ['WN', 'ADG', 'FIC', 'MultinerdIT']
    for test_name in test_datasets_names:
        test_set = load_dataset("json", data_files=f'/nfsd/VFdisk/zamaiandre/SLIMER_IT/src/SOTA/GNER/data/{test_name}/test_GNER_format.json')['train']

        test_set = test_set.to_list()
        inputs = []
        for sample_GNER in test_set:
            # For LLaMA Model, instruction part are wrapped with [INST] tag
            input_texts = f"[INST] {sample_GNER['instance']['instruction_inputs']} [/INST]"
            inputs.append(input_texts)

        responses = vllm_model.generate(inputs, sampling_params)
        for i, response in enumerate(responses):
            response = response.outputs[0].text
            # response = response[response.find("[/INST]") + len("[/INST]"):].strip()
            # print(response)
            test_set[i]['prediction'] = response
            if i == 0:
                print(test_set[0])
                sys.stdout.flush()

        test_set = Dataset.from_list(test_set)

        test_set.to_json(f'./predictions/GNER-IT-vllm/{test_name}.jsonl')


if __name__ == "__main__":
    main()