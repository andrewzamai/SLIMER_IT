<div align="center">
  <h1>👻 SLIMER: Show Less Instruct More Entity Recognition</h1>
</div>


<p align="center">
    <a href="https://github.com/andrewzamai/SLIMER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-Apache2.0-blue"></a>
    <a href="https://huggingface.co/expertai/SLIMER"><img alt="Models" src="https://img.shields.io/badge/🤗-Models-green"></a>
    <a href="https://arxiv.org/abs/2407.01272"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://www.expert.ai/"><img src="https://img.shields.io/badge/company-expert.ai-blueviolet"></a>
</p>

## Instruct your LLM with Definitions and Guidelines for Zero-Shot NER 🔎 📖

Designed to work on:

&nbsp;&nbsp;&nbsp;&nbsp;✅ Out-Of-Domain inputs (e.g. news, science, politics, music ...)

&nbsp;&nbsp;&nbsp;&nbsp;✅ Never-Seen-Before Named Entities (the model was not trained on that entity type? It will tag it anyway!)

<div align="center">
<img src="assets/SLIMER_prompt.png" alt="Alt text" style="max-width: 100%; width: 275px;">
</div>


## 📄 TL;DR
Existing models primarily focus on addressing zero-shot NER on Out-of-Domain inputs, while fine-tuning on an extensive number of entity classes that often highly or completely overlap with the test sets used in the literature. 

But what happens in everyday use when they encounter entities that were not included in training? 
As demonstrated on BUSTER, they are likely to significantly underperform 📉, revealing a critical weakness in handling never-seen-before entity types. 🆘

In this work, we propose SLIMER, an approach designed to tackle never-seen-before entity tags by instructing the model on fewer examples from a reduced tag set, and by leveraging a prompt enriched with definition and guidelines.

Experiments demonstrate that definition and guidelines yield to faster, more robust learning, and better performance, particularly when labelling unseen named entities. 

When compared to other existing solutions under similar training conditions, SLIMER demonstrates superior capability in handling unseen entities, delivering more robust performance in real-world applications. 🏆

PROs:

&nbsp;&nbsp;&nbsp;&nbsp;✅ guide your LLM with external knowledge about the NE to tag 
&nbsp;&nbsp;&nbsp;&nbsp;✅ definition and guidelines simple syntax (no code)

&nbsp;&nbsp;&nbsp;&nbsp;✅ flexibility to different annotation schemes 
&nbsp;&nbsp;&nbsp;&nbsp;✅ granularity and exceptions (all people not musicians)

&nbsp;&nbsp;&nbsp;&nbsp;✅ disambiguate on polysemous NEs
&nbsp;&nbsp;&nbsp;&nbsp;✅ nested-NER (one span of text, multiple categories)

&nbsp;&nbsp;&nbsp;&nbsp;✅ long documents handling

CONs:

&nbsp;&nbsp;&nbsp;&nbsp;❌ does not scale well with increasing label set cardinality (future work: prefix-caching)


## Installation

You will need to install the following dependencies to run SLIMER:
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## Running

Evaluate SLIMER w/ D&G on MIT/CrossNER/BUSTER
```
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER --with_guidelines
```

Train, merge, evaluate your SLIMER:
```
# 1) train on PileNER-subset with Definition and Guidelines, 391 NEs, 5 samples per NE
PYTHONPATH=$(pwd) python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines

# 2) merge LORA weights
PYTHONPATH=$(pwd) python src/SFT_finetuning/commons/merge_lora_weights.py 391 5 5 --with_guidelines

# 3) evaluate SLIMER model on MIT/CrossNER/BUSTER
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py LLaMA2_7B_5pos_5neg_perNE_top391NEs_TrueDef --with_guidelines
```

## Run it on your NER data!

Running SLIMER on your data is simple as:

1) implement *load_datasetdict_BIO()* (tell where and how to load your NER data), *get_map_to_extended_NE_name()* (e.g. PER-->PERSON) of **Data_Interface** abstract class
   
2) provide your Definition and Guidelines for each NE class
   
3) run SLIMER!

## Demo usage

A simple inference example is as follows:

```python
from vllm import LLM, SamplingParams
from src.SFT_finetuning.commons.prompter import SLIMER_instruction_prompter, Prompter


vllm_model = LLM("expertai/SLIMER")
# it is recommended to use a temperature of 0
# max_new_tokens can be adjusted depending on the expected length and number of entities (default 128)
sampling_params = SamplingParams(temperature=0, max_tokens=128, stop=['</s>'])

# suppose we want to extract the entities of type "algorithm", we just need to write the definition and guidelines in simple syntax
tag_to_extract = "algorithm"
tag_definition = "ALGORITHM entities refer to specific computational procedures or methods designed to solve a problem or perform a task within the field of computer science or related disciplines."
tag_guidelines = "Avoid labeling generic technology or software names without specific algorithmic context. Exercise caution with terms that may denote both a specific algorithm and a generic concept, such as 'neural network'."

# format the Def & Guidelines into SLIMER instruction
slimer_prompter = SLIMER_instruction_prompter("SLIMER_instruction_template", template_path='./src/SFT_finetuning/templates')
instruction = slimer_prompter.generate_prompt(ne_tag=tag_to_extract, definition=tag_definition, guidelines=tag_guidelines)
print(instruction)
"Extract the Named Entities of type ALGORITHM from the text chunk you have read. You are given a DEFINITION and some GUIDELINES.\nDEFINITION: ALGORITHM entities refer to specific computational procedures or methods designed to solve a problem or perform a task within the field of computer science or related disciplines.\nGUIDELINES: Avoid labeling generic technology or software names without specific algorithmic context. Exercise caution with terms that may denote both a specific algorithm and a generic concept, such as 'neural network'.\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present."

input_text = "Typical generative model approaches include naive Bayes classifier s , Gaussian mixture model s , variational autoencoders and others ."

# prefix the input text to the instruction and format it into LLaMA-2 template 
llama2_prompter = Prompter('LLaMA2-chat', template_path='./src/SFT_finetuning/templates', eos_text='')
prompts = [llama2_prompter.generate_prompt(instruction, input_text)]
print(prompts[0])
"[INST] You are given a text chunk (delimited by triple quotes) and an instruction.\nRead the text and answer to the instruction in the end.\n\"\"\"\nTypical generative model approaches include naive Bayes classifier s , Gaussian mixture model s , variational autoencoders and others .\n\"\"\"\nInstruction: Extract the Named Entities of type ALGORITHM from the text chunk you have read. You are given a DEFINITION and some GUIDELINES.\nDEFINITION: ALGORITHM entities refer to specific computational procedures or methods designed to solve a problem or perform a task within the field of computer science or related disciplines.\nGUIDELINES: Avoid labeling generic technology or software names without specific algorithmic context. Exercise caution with terms that may denote both a specific algorithm and a generic concept, such as 'neural network'.\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present.\n[/INST]"

responses = vllm_model.generate(prompts, sampling_params)
all_pred_answers = [output.outputs[0].text.strip() for output in responses]
print(all_pred_answers[0])
"[\"naive Bayes classifier\", \"Gaussian mixture model\", \"variational autoencoders\"]"
```
    
## 📚 Citation

If you find SLIMER useful in your work or research, please consider citing our paper:

```bibtex
@misc{zamai2024lessinstructmoreenriching,
      title={Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER}, 
      author={Andrew Zamai and Andrea Zugarini and Leonardo Rigutini and Marco Ernandes and Marco Maggini},
      year={2024},
      eprint={2407.01272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01272}, 
}
```
