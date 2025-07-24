from typing import Any, Dict, List, Type

from q2d.common.types import SubscriptableBaseModel
from q2d.common.utils import apply_template, seed_everything
from q2d.generation.generate_utils import InferenceEngine, InferenceEngineConfig
from q2d.generation.prompts import BasePrompt


class Config(SubscriptableBaseModel):
    prompt: BasePrompt
    engine: Type[InferenceEngine]
    engine_config: InferenceEngineConfig
    seed: int = 42


def generate(input_dataset: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    seed_everything(config.seed)

    messages_batch = [
        apply_template(
            system_prompt_template=config.prompt.system_prompt_template,
            user_prompt_template=config.prompt.user_prompt_template,
            user_prompt_kwargs=data,
        )
        for data in input_dataset
    ]

    engine = config.engine(config.engine_config)
    generated_outputs = engine.generate_batched(messages_batch)

    generated_output_json = [
        {
            **data,
            "output": output.get_output(),
        }
        for data, output in zip(input_dataset, generated_outputs)
    ]

    return generated_output_json
