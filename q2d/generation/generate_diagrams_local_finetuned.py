import json
from typing import get_args

from q2d.common.types import QUERY_VERSIONS, Graph
from q2d.generation.generate_api import Config, generate
from q2d.generation.generate_utils import AsyncOpenAIEngine, OpenAIEngineConfig
from q2d.generation.prompts import DiagramsFinetunedPrompt

ConfigFinetuned = Config(
    seed=1111,
    prompt=DiagramsFinetunedPrompt,
    engine=AsyncOpenAIEngine,
    engine_config=OpenAIEngineConfig(
        model_url="http://127.0.0.1:8000/v1",
        api_key="empty",
        retries_num=3,
        response_format=Graph,
        model_params={
            "model": "finetuned_model",
            "temperature": 0.0,
            "extra_body": {"guided_decoding_backend": "outlines"},
        },
    ),
)
if __name__ == "__main__":
    input_path = "./datasets/test.json"
    output_path = "./datasets/finetuned_model.json"

    with open(input_path, "r") as f:
        input_dataset = json.load(f)

    input_dataset = [data | {"version": version} for data in input_dataset for version in get_args(QUERY_VERSIONS)]

    generated_outputs = generate(input_dataset, ConfigFinetuned)
    output_dataset = []
    for i in range(len(generated_outputs)):
        data = {
            field: generated_outputs[i][field] for field in ("language", "code", "repo", "path", "query", "version")
        }
        data["diagram"] = generated_outputs[i]["output"]
        output_dataset.append(data)

    with open(output_path, "w") as f:
        json.dump(output_dataset, f, indent=2)
