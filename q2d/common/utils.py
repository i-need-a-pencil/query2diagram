import hashlib
import os
import random
from string import Formatter
from typing import Dict, List, Optional

import numpy as np
import torch


class OptionalFormatter(Formatter):
    def get_value(self, key, args, kwds):
        try:
            return super().get_value(key, args, kwds)
        except KeyError:
            return f"{key}"


FORMATTER = OptionalFormatter()


def apply_template(
    system_prompt_template: Optional[str],
    user_prompt_template: Optional[str],
    user_prompt_kwargs: Dict[str, str],
    assistant_prompt_template: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages = []
    if system_prompt_template is not None:
        messages += [{"role": "system", "content": system_prompt_template}]

    if user_prompt_template is not None:
        messages += [{"role": "user", "content": FORMATTER.format(user_prompt_template, **user_prompt_kwargs)}]

    if assistant_prompt_template is not None:
        messages += [{"role": "assistant", "content": assistant_prompt_template}]
    return messages


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data_id(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()
