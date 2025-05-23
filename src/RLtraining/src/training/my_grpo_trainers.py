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

from peft import PeftModel
import os

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def get_hint_probability(
    input_len: int,
    gold_diagnosis: str,
    min_len: int = 300,
    max_len: int = 600,
    min_prob: float = 0.4,
    max_prob: float = 0.8,
) -> float:
    """
    Computes the probability of appending a hint to the prompt based on input length
    and diagnosis type using a cosine-based schedule.
    """
    # Clamp length
    input_len = max(min_len, min(input_len, max_len))
    progress = (input_len - min_len) / (max_len - min_len)  # Normalize to [0, 1]
    cosine = math.cos(progress * math.pi)
    scaling_factor = (1.0 - cosine) / 2.0  # Scales from 0 (short) to 1 (long)

    if gold_diagnosis == "CN":
        # CN: low prob if short, increase with length
        prob = min_prob + (max_prob - min_prob) * scaling_factor
    else:
        # Dementia: high prob if short, decrease with length
        prob = max_prob - (max_prob - min_prob) * scaling_factor

    return prob


def suffix_to_last_message(prompt: list[dict], txt: str) -> list[dict]:
    """
    Appends a suffix `txt` to the 'content' field of the last message in a prompt list.
    """
    if not prompt:
        raise ValueError("Prompt list is empty.")
    if "content" not in prompt[-1]:
        raise ValueError("Last message in prompt does not contain 'content' key.")
    prompt = prompt.copy() # avoid in-place modification
    prompt[-1] = prompt[-1].copy()
    prompt[-1]["content"] += txt
    return prompt


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

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample
    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> print_prompt_completions_sample(prompts, completions, rewards, 42)
    ╭────────────────────── Step 42 ───────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │ │
    │ └────────────┴──────────────┴─────────────┴────────┘ │
    ╰──────────────────────────────────────────────────────╯
    ```
    """
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("gold_diagnosis", style="bright_yellow")
    table.add_column("completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}

    for i in range(len(prompts)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]  # 2 decimals
        prompt_text = str(prompts[i]) if prompts[i] is not None else ""
        completion_text = str(completions[i]) if completions[i] is not None else ""
        table.add_row(Text(prompt_text), Text(completion_text), *reward_values)
        #table.add_row(Text(prompts[i]), Text(completions[i]), *reward_values)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class MyCustomGRPOTrainer(GRPOTrainer):

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        #### PARAMS, todo: to be passed in __init__
        #max_hint_prob = 0.8
        #num_hinted_compl_per_prompt = 2
        max_hint_prob = 0
        num_hinted_compl_per_prompt = 0

        prompts = [x["prompt"] for x in inputs]
        # NB: assuming some existing dataset columns
        # NB: assumes txt_report_len field pre-computed (number of tokens MRI report)
        gold_diagnoses = [x["gold_diagnosis"] for x in inputs]
        txt_report_lengths = [x["txt_report_len"] for x in inputs]
        subjects = [x["subject"] for x in inputs]
        
        # apply chat template, tokenize and pad to compute prompt_ids, prompt_mask
        # NB: we don't leave any traces a prompt may have been hinted
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs) # use Trainer method, otherwise would use GRPOTrainer method wrongly
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts = gather_object(prompts) # not formatted into chat-template yet
            all_gold_diagnoses = gather_object(gold_diagnoses)
            all_txt_report_lengths = gather_object(txt_report_lengths)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one.
                ordered_set_of_prompts = all_prompts[:: self.num_generations]
                ordered_set_of_gold_diagnoses = all_gold_diagnoses[:: self.num_generations]
                ordered_set_of_all_txt_report_lengths = all_txt_report_lengths[:: self.num_generations]
                
                standard_prompts_set = []
                hinted_prompts_set = []
                mode = "eval" if self.control.should_evaluate else "train"
                for prompt, gold_diagnosis, input_len in zip(ordered_set_of_prompts, ordered_set_of_gold_diagnoses, ordered_set_of_all_txt_report_lengths):
                    # should_hint = random.random() < 1  # 80% chance
                    hint_prob = get_hint_probability(input_len, gold_diagnosis, max_prob=max_hint_prob)
                    should_hint = random.random() < hint_prob
                    if should_hint and mode == 'train' and max_hint_prob != 0:
                        suffix = f"\nHint: formulate your reasoning considering {gold_diagnosis} as the most likely diagnosis."
                        hinted_prompt = suffix_to_last_message(prompt, suffix)
                        print(f"--> Hinting for class {gold_diagnosis}, input_len: {input_len} hint_prob: {hint_prob:.2f}")
                    else:
                        hinted_prompt = prompt  # No suffix, hint is identical to the original
                    standard_prompts_set.append(prompt)       # For n-1 completions
                    hinted_prompts_set.append(hinted_prompt)  # For 1 completion
                # apply chat template
                standard_prompts_set = [maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in standard_prompts_set]
                hinted_prompts_set = [maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in hinted_prompts_set]

                #print(standard_prompts_set)
                #print(hinted_prompts_set)

                with profiling_context(self, "vLLM.generate"):
                    std_completions = self.vllm_client.generate(
                        prompts=standard_prompts_set,
                        n=self.num_generations - num_hinted_compl_per_prompt,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )

                    if num_hinted_compl_per_prompt != 0:
                        hint_completions = self.vllm_client.generate(
                            prompts=hinted_prompts_set,
                            n=num_hinted_compl_per_prompt,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                    
                    num_prompts = len(standard_prompts_set)
                    #print(num_prompts)
                    num_std = self.num_generations - num_hinted_compl_per_prompt
                    
                    completion_ids = []
                    for i in range(num_prompts):
                        std_for_prompt = std_completions[i * num_std : (i + 1) * num_std]
                        if num_hinted_compl_per_prompt != 0:
                            hint_for_prompt = hint_completions[i * num_hinted_compl_per_prompt : (i + 1) * num_hinted_compl_per_prompt]
                            #print(f"DEBUG: For prompt {i}: standard completions count: {len(std_for_prompt)}")
                            #print(f"DEBUG: For prompt {i}: hint completion type: {type(hint_for_prompt)}, length: {len(hint_for_prompt)}")
                            combined = std_for_prompt + hint_for_prompt
                            #print(f"DEBUG: For prompt {i}: combined completions count: {len(combined)}")
                            completion_ids.extend(combined)
                        else:
                            completion_ids.extend(std_for_prompt)
                    #print(f"DEBUG: Total aggregated completions: {len(completion_ids)} (Expected: {num_prompts * self.num_generations})")
            else:
                completion_ids = [None] * len(all_prompts)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None #Please read thoughtful this code, understand it and return the keypoints it implements. Focus on how a prompt generates N completions and the advantages are computed. here is the code 

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        """
        # HERE TRL implementation computes rewards within each process and then gathers rewards for group computation
        # we aggregate before to compute rewards on gathered processes, such that we have all completions for each prompt together and not across processes

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        """

        # First gather all prompts and completions across processes
        gathered_prompts = gather_object(prompts)
        gathered_completions = gather_object(completions)
        # Gather all input columns except "prompt" and "completion"
        gathered_reward_kwargs = {}
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        for key in keys:
            # Extract values for this key from inputs
            values = [example[key] for example in inputs]
            # Gather values across processes
            gathered_values = gather_object(values)
            gathered_reward_kwargs[key] = gathered_values

        if self.accelerator.is_main_process:
            
            rewards_per_func = torch.zeros(len(gathered_prompts), len(self.reward_funcs), device=device)

            for i, (reward_func, _) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Skip the nn.Module case as requested
                if not isinstance(reward_func, nn.Module):
                    reward_func_name = reward_func.__name__
                    with profiling_context(self, reward_func_name):
                        # Now use the gathered data to compute rewards across all examples at once
                        output_reward_func = reward_func(
                            prompts=gathered_prompts, 
                            completions=gathered_completions, 
                            **gathered_reward_kwargs
                        )
                        
                        # Convert None values to NaN
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            if torch.isnan(rewards_per_func).all(dim=1).any():
                nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in gathered_reward_kwargs.items()}
                row_reward_kwargs["prompt"] = gathered_prompts[nan_row_idx]
                row_reward_kwargs["completion"] = gathered_completions[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )

            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)

            #print(advantages.shape)
        else:
            # Placeholder tensor to receive broadcast
            advantages = torch.empty(len(gathered_prompts), device=device, dtype=torch.float32)
            mean_grouped_rewards = torch.empty(len(gathered_prompts), device=device, dtype=torch.float32)
            std_grouped_rewards = torch.empty(len(gathered_prompts), device=device, dtype=torch.float32)
            rewards_per_func = torch.empty(len(gathered_prompts), len(self.reward_funcs), device=device, dtype=torch.float32)

        advantages = broadcast(advantages, from_process=0)
        mean_grouped_rewards = broadcast(mean_grouped_rewards, from_process=0)
        std_grouped_rewards = broadcast(std_grouped_rewards, from_process=0)
        rewards_per_func = broadcast(rewards_per_func, from_process=0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        #mean_grouped_rewards = mean_grouped_rewards[process_slice]
        #std_grouped_rewards = std_grouped_rewards[process_slice]
        #rewards_per_func = rewards_per_func[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Get the names of the reward functions
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(gold_diagnoses)
            #prompts_to_log = gather_object(prompts_text)
            #completions_to_log = gather_object(txt_report_lengths)
            completions_to_log = gather_object(completions_text)
            #rewards_to_log = {
            #    reward_func_name: rewards_per_func[:, i] for i, reward_func_name in enumerate(reward_func_names)
            #}
            rewards_to_log = {}

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                        self.num_completions_to_print,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    # wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        #loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        #loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length) # DRGRPO

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss






class MyCustomWeightedGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        
        # Ensure inputs is not empty before accessing inputs[0]
        samples_weights = None
        if inputs and "samples_weights" in inputs[0]:
            # print("Found samples_weights in inputs :)")
            samples_weights = [x["samples_weights"] for x in inputs]
            samples_weights = torch.tensor(samples_weights, dtype=torch.float32, device=self.accelerator.device)

        # Call the original method to get the base output
        output = super()._generate_and_score_completions(inputs)
        # dictionary of torch Tensors
        #output = return {
        #    "prompt_ids": prompt_ids,  
        #    "prompt_mask": prompt_mask,
        #    "completion_ids": completion_ids,
        #    "completion_mask": completion_mask,
        #    "old_per_token_logps": old_per_token_logps,
        #    "ref_per_token_logps": ref_per_token_logps,
        #    "advantages": advantages,
        #}
        
        # Add samples_weights if it exists
        if samples_weights is not None:
            output["samples_weights"] = samples_weights
       
        return output

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # prev: loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        samples_weights = inputs.get("samples_weights", None)  # Shape: [batch_size] or None
        if samples_weights is not None:
            #print(samples_weights.shape)
            samples_weights = samples_weights.unsqueeze(1)  # Shape: [batch_size, 1] for broadcasting
        else:
            samples_weights = torch.ones_like(per_token_loss[:, 0]).unsqueeze(1)  # Default to uniform weights

        # Apply weights to per-sample loss before summing
        weighted_per_token_loss = per_token_loss * completion_mask * samples_weights
        #print(weighted_per_token_loss)

        # Compute weighted loss
        loss = weighted_per_token_loss.sum() / (completion_mask * samples_weights).sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss