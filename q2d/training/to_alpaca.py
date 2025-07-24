import json
import random
from typing import Any, Dict, List

from q2d.common.utils import get_data_id, seed_everything
from q2d.generation.prompts import BasePrompt, DiagramsFinetunedPrompt


class Config:
    input_path = "./datasets/claude_sonnet_synth_fixed_final.json" # "./datasets/claude_sonnet_synth.json"
    sampled_ids_path = "./datasets/sampled_ids.json"
    output_path = "./datasets/training/diagrams_alpaca.json"
    output_eval_path = "./datasets/training/diagrams_alpaca_eval.json"
    seed = 1111


def convert_to_alpaca(dataset: List[Dict[str, Any]], prompt_template: BasePrompt) -> List[Dict[str, Any]]:
    return [
        {
            "system": prompt_template.system_prompt_template,  # system prompt (optional)
            "instruction": prompt_template.user_prompt_template.format(
                code=data["code"], query=data["query"], version=data["version"]
            ),  # human instruction (required)
            "output": str(data["diagram"]),  # model response (required)
        }
        for data in dataset
    ]


if __name__ == "__main__":
    seed_everything(Config.seed)
    with open(Config.sampled_ids_path, "r") as f:
        validation_ids = set(json.load(f)["validation"])

    with open(Config.input_path, "r") as f:
        dataset = json.load(f)

    train_dataset = []
    val_dataset = []
    for i, data in enumerate(dataset):
        data_id = get_data_id(data["code"])
        if data_id in validation_ids:
            val_dataset.append(data)
        else:
            train_dataset.append(data)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    converted_train_dataset = convert_to_alpaca(train_dataset, DiagramsFinetunedPrompt)
    random.shuffle(converted_train_dataset)
    with open(Config.output_path, "w") as f:
        json.dump(converted_train_dataset, f, indent=2)

    converted_eval_dataset = convert_to_alpaca(val_dataset, DiagramsFinetunedPrompt)
    with open(Config.output_eval_path, "w") as f:
        json.dump(converted_eval_dataset, f, indent=2)
