import json

import numpy as np
from q2d.common.utils import get_data_id, seed_everything


class Config:
    input_path = "./datasets/claude_sonnet_synth_fixed_final.json"
    output_path = "./datasets/sampled_ids.json"
    seed = 1111


if __name__ == "__main__":
    seed_everything(Config.seed)

    with open(Config.input_path, "r") as f:
        dataset = json.load(f)

    languages = {data["language"] for data in dataset}
    assert len(languages) == 12

    sampled_ids = []

    for lang in languages:
        subset = [data for data in dataset if data["language"] == lang]
        chosen_val_data = np.random.choice(subset, 1)[0]

        sampled_ids.append(get_data_id(code=chosen_val_data["code"]))

    with open(Config.output_path, "w") as f:
        json.dump({"validation": sampled_ids}, f)
