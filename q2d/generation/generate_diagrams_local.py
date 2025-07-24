import json
import re
from ast import literal_eval

from q2d.common.types import Graph, SubscriptableBaseModel
from q2d.generation.generate_api import Config, generate
from q2d.generation.generate_utils import OpenAIEngine, OpenAIEngineConfig
from q2d.generation.prompts import DiagramsZeroShotPrompt

FIELD_TO_VERSION = {
    "minimal_version": "minimal",
    "medium_version": "medium",
    "full_version": "full",
}


class Output(SubscriptableBaseModel):
    minimal_version: Graph
    medium_version: Graph
    full_version: Graph
    text_answer: str

    @staticmethod
    def convert_from_string(string: str) -> "Output":
        string_clean = re.sub(r"(?<![a-zA-Z0-9_])null(?![a-zA-Z0-9_])", "None", string.strip())
        converted_dict = literal_eval(string_clean)
        return Output(**converted_dict)


ConfigFinetuned = Config(
    seed=1111,
    prompt=DiagramsZeroShotPrompt,
    engine=OpenAIEngine,
    engine_config=OpenAIEngineConfig(
        model_url="http://127.0.0.1:8000/v1",
        api_key="empty",
        retries_num=3,
        response_format=Output,
        stream=True,
        model_params={
            "model": "Qwen2.5-Coder-14B-Instruct-bnb-4bit",
            "temperature": 0.0,
            "extra_body": {"guided_decoding_backend": "outlines"},
        },  # model params
    ),
)
if __name__ == "__main__":
    input_path = "./datasets/test.json"
    output_path = "./datasets/Qwen_base.json"

    with open(input_path, "r") as f:
        input_dataset = json.load(f)

    generated_outputs = generate(input_dataset, ConfigFinetuned)
    output_dataset = []
    for i in range(len(generated_outputs)):
        try:
            output = Output.convert_from_string(generated_outputs[i]["output"])

            data = {field: generated_outputs[i][field] for field in ("language", "code", "repo", "path", "query")}
            data["text_answer"] = output["text_answer"]
            for field in ("minimal_version", "medium_version", "full_version"):
                output_dataset.append(data | {"version": FIELD_TO_VERSION[field], "diagram": str(output[field])})
        except (ValueError, TypeError, AttributeError, SyntaxError) as e:
            print(f"{i}: {str(e)}")
            data = {field: generated_outputs[i][field] for field in ("language", "code", "repo", "path", "query")}
            data["text_answer"] = None
            data["version"] = None
            data["diagram"] = None
            data["output"] = generated_outputs[i]["output"]

            output_dataset.append(data)

    with open(output_path, "w") as f:
        json.dump(output_dataset, f, indent=2)
