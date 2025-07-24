import json
from typing import Optional

from q2d.generation.generate_api import Config, generate
from q2d.generation.generate_utils import AsyncOpenAIEngine, OpenAIEngineConfig
from q2d.generation.prompts import DiagramsUserOnlyZeroShotPrompt
from sklearn.model_selection import train_test_split


def extract_output(content: Optional[str]) -> Optional[str]:
    if content is None:
        return None

    if not ("<think>" in content and "</think>" in (content := content.rsplit("<think>", 1)[1])):
        return None

    if not ("<candidates>" in content and "</candidates>" in (content := content.rsplit("<candidates>", 1)[1])):
        return None

    if not ("<final_output>" in content and "</final_output>" in (content := content.rsplit("<final_output>", 1)[1])):
        return None

    output = content.rsplit("</final_output>", 1)[0].strip()
    if len(output) == 0:
        return None
    return output


ConfigFinetuned = Config(
    seed=1111,
    prompt=DiagramsUserOnlyZeroShotPrompt,
    engine=AsyncOpenAIEngine,
    engine_config=OpenAIEngineConfig(
        model_url="http://127.0.0.1:8000/v1",
        api_key="empty",
        retries_num=3,
        retry_after=10,
        stream=True,
        model_params={
            "model": "DeepSeekR1-70B",
            "temperature": 0.6,
            "top_p": 0.9,
        },  # model params
    ),
)
if __name__ == "__main__":
    input_path = "./datasets/sampled.json"
    output_path = "./datasets/questions_r1.json"
    subset_size = 10000

    with open(input_path, "r") as f:
        input_dataset = json.load(f)

    input_dataset = train_test_split(
        input_dataset,
        train_size=subset_size,
        random_state=Config.seed,
        shuffle=True,
        stratify=[data["language"] for data in input_dataset],
    )[0]

    generated_outputs = generate(input_dataset, ConfigFinetuned)
    output_dataset = []
    for i in range(len(generated_outputs)):
        data = {field: generated_outputs[i][field] for field in ("language", "code", "repo", "path")}
        data["user_queries"] = generated_outputs[i]["extracted_output"]
        data["generated_output"] = generated_outputs[i]["raw_output"]

        output_dataset.append(data)

    with open(output_path, "w") as f:
        json.dump(output_dataset, f, indent=2)
