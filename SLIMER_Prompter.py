"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class SLIMER_Prompter(object):

    __slots__ = ("template", "template_path", "_verbose")

    def __init__(self, template_name: str = "", template_path: str = "templates", verbose: bool = True):
        self._verbose = verbose
        if not template_name:
            template_name = "SLIMER_instruction_it"
        file_name = osp.join(template_path, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}\n")

    def generate_prompt(
        self,
        ne_tag: str,
        definition: Union[None, str] = None,
        guidelines: Union[None, str] = None
    ) -> str:
        if definition and guidelines:
            res = self.template["with_DeG"].replace(
                "{ne_tag}", ne_tag).replace(
                "{definition}", definition).replace(
                "{guidelines}", guidelines)
        else:
            res = self.template["without_DeG"].format(
                ne_tag=ne_tag
            )
        return res


if __name__ == '__main__':

    prompt = SLIMER_Prompter("SLIMER_instruction_it", template_path="templates").generate_prompt(ne_tag='PERSONA') #, definition='Questa è la definizione.', guidelines="Queste sono linee guida.")
    print(prompt)