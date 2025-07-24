import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from q2d.datasets.nearest_neighbours import (
    _ID_TYPE,
    _NEAREST_SUFFIX,
    compute_near_duplicates,
)
from tqdm.auto import tqdm


class Config:
    input_file_path = "./datasets/extracted_files.json"
    output_file_path_deduped = "./datasets/deduped.json"
    output_file_path_sampled = "./datasets/sampled.json"

    target_languages = {
        "Python",
        "Java",
        "Go",
        "JavaScript",
        "C++",
        "TypeScript",
        "PHP",
        "C",
        "C#",
        "Rust",
        "Scala",
        "Kotlin",
    }

    top_priority_languages = {
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "TypeScript",
        "C",
    }
    min_repo_size = 25
    target_size = 100000
    min_length = 3000
    max_length = 15000
    seed = 1111


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def remove_leading_comments(file_content: str, language: str, max_leading_comments: int = 50) -> str:
    language_patterns = {
        "Python": [r"^#.*", r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'],
        "Java": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Go": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "JavaScript": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "C++": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "TypeScript": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "PHP": [r"^//.*", r"^#.*", r"^/\*[\s\S]*?\*/"],
        "Ruby": [r"^#.*", r"^=begin[\s\S]*?=end"],
        "C": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "C#": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Rust": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Scala": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Kotlin": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Swift": [r"^//.*", r"^/\*[\s\S]*?\*/"],
        "Perl": [r"^#.*"],
        "Haskell": [r"^--.*", r"^\{-[\s\S]*?-\}"],
        "R": [r"^#.*"],
        "Shell": [r"^#.*"],
    }

    # Skip common headers
    head = ""
    file_content = file_content.strip()
    if file_content[:2] == "#!" or file_content[:6] == "<\\?php":
        file_content_split = file_content.split("\n", 1)
        if len(file_content_split) == 1:
            return file_content
        head = file_content_split[0].strip()
        file_content = file_content_split[1].strip()

    comment_patterns = language_patterns[language]
    combined_pattern = "|".join(f"({pattern})" for pattern in comment_patterns)

    regex = re.compile(f"^(?:{combined_pattern})[\\s\n]*", re.DOTALL)

    previous_content = None
    while previous_content != file_content and max_leading_comments > 0:
        previous_content = file_content

        match = regex.match(file_content)
        if match:
            file_content = file_content[match.end() :].lstrip()
            max_leading_comments -= 1
        else:
            break

    return head + "\n" + file_content


def non_english_keyboard() -> str:
    # additional legal unicode symbols: 一–━¬⎯±−…→О\’
    return r"[^a-zA-Z0-9\s\.\/<>?;:\"\'`\!@#$%\^&\*\(\)\[\]\{\}_+=\|\\\-~\,一]"


def deduplicate_dataset(
    dataset: List[Dict[str, Any]], clone_field: str, batch_size: int = 10_000, threshold: float = 0.9
) -> List[Dict[str, Any]]:
    dataset_with_ids: Dict[_ID_TYPE, Dict[str, Any]] = {i: data for i, data in enumerate(dataset)}

    compute_near_duplicates(dataset_with_ids, clone_field, batch_size, threshold)

    similarity_field = clone_field + _NEAREST_SUFFIX
    added_keys = set()
    deduplicated_dataset = dict()

    for key, value in tqdm(dataset_with_ids.items(), desc="remove duplicates..."):
        if similarity_field not in value or all(pair_key not in added_keys for pair_key, _ in value[similarity_field]):
            deduplicated_dataset[key] = {k: v for k, v in value.items() if k != similarity_field}
            added_keys.add(key)
    removed_num = len(dataset_with_ids) - len(deduplicated_dataset)
    print(
        f"Removed {removed_num} out of {len(dataset_with_ids)} or {removed_num/len(dataset_with_ids)*100:.2f}% from dataset"
    )

    return list(deduplicated_dataset.values())


if __name__ == "__main__":
    seed_everything(Config.seed)

    with open(
        Config.input_file_path,
        "r",
    ) as f:
        dataset = json.load(f)
    repos_files_num = Counter(
        data["project_root"].rsplit("/", 2)[-2] + "/" + data["project_root"].rsplit("/", 2)[-1] for data in dataset
    )
    restricted_repos = {repo for repo, freq in repos_files_num.items() if freq < Config.min_repo_size}

    code_filters = [
        ("code is not None", lambda data: data["code"] is not None),
        ("code >= min_length", lambda data: len(data["code"]) >= Config.min_length),
        ("code <= max_length", lambda data: len(data["code"]) <= Config.max_length),
        (
            "only english keyboard symbols in code",
            lambda data: len(data["code"]) == 0 or not (re.search(non_english_keyboard(), data["code"])),
        ),
    ]

    transformed_data = [
        data
        | {
            "code": None if data["code"] is None else remove_leading_comments(data["code"].strip(), data["language"]),
            "repo": data["project_root"].rsplit("/", 2)[-2] + "/" + data["project_root"].rsplit("/", 2)[-1],
        }
        for data in tqdm(dataset)
    ]

    subset_to_sample = [
        data
        for data in tqdm(transformed_data)
        if data["repo"] not in restricted_repos
        and data["language"] in Config.target_languages
        and all(filter_func(data) for _, filter_func in code_filters)
    ]
    print(f"Filtered data size: {len(subset_to_sample)}")

    subset_to_sample = deduplicate_dataset(subset_to_sample, "code")
    print(f"Deduped filtered data size: {len(subset_to_sample)}")

    with open(Config.output_file_path_deduped, "w") as f:
        json.dump(subset_to_sample, f)

    subset = []
    size_by_language = Config.target_size // len(Config.target_languages) // 2
    size_for_top_priority_languages = size_by_language + Config.target_size // len(Config.top_priority_languages) // 2

    for language in Config.target_languages:
        language_subset_to_sample = [data for data in subset_to_sample if data["language"] == language]
        cur_lang_size = (
            size_for_top_priority_languages if language in Config.top_priority_languages else size_by_language
        )
        if len(language_subset_to_sample) <= cur_lang_size:
            subset += language_subset_to_sample
        else:
            subset += np.random.choice(
                language_subset_to_sample,
                size=cur_lang_size,
                replace=False,
            ).tolist()

    print(f"Sampled data size: {len(subset)}")

    with open(Config.output_file_path_sampled, "w") as f:
        json.dump(subset, f)
