import json
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class DownloadStatus(Enum):
    Done = 0
    Error = -1


def parse_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="unicode_escape") as file:
                code = file.read().strip()
        except UnicodeDecodeError:
            return None
    return code


def traverse_project(
    project_root: str, extension_to_language: Dict[str, str]
) -> Tuple[Optional[List[Dict[str, str]]], Tuple[DownloadStatus, str]]:
    result = []
    try:
        for root, _, files in os.walk(project_root):
            for file in files:
                file_ext = os.path.splitext(file)[1]
                file_path = os.path.join(root, file)

                if file_ext in extension_to_language and os.path.isfile(file_path):
                    file_data = parse_file(file_path=file_path)
                    result.append(
                        {
                            "code": file_data,
                            "path": file_path,
                            "project_root": project_root,
                            "language": extension_to_language[file_ext],
                        }
                    )
        return result, (DownloadStatus.Done, "ok")
    except Exception as e:
        return None, (DownloadStatus.Error, f"{str(type(e))}: {str(e)}")


def get_repo_path(main_repo_dir: str, repo_url: str) -> str:
    git_url = repo_url.rsplit(":", 1)[0] if ":" in repo_url[6:] else repo_url

    repo_maintainer = git_url.rsplit("/", 2)[1]
    repo_name = git_url.rsplit("/", 2)[2]
    return os.path.join(main_repo_dir, repo_maintainer, repo_name)


class Config:
    lang_extensions = "./datasets/lang_extensions.json"
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
    input_file = "./datasets/top_150.csv"
    main_repo_dir = "./datasets/diagrams-repos"
    output_path = "./datasets/extracted_files.json"


def main():
    with open(Config.lang_extensions, "r") as f:
        lang_extensions = json.load(f)
        extension_to_language = {
            ext: d["name"]
            for d in lang_extensions
            if d["name"] in Config.target_languages and "extensions" in d
            for ext in d["extensions"]
        }

    dataset = pd.read_csv(Config.input_file)
    repos = [data["repo"] for data in dataset.to_dict(orient="records")]
    repos_paths = [get_repo_path(Config.main_repo_dir, repo) for repo in repos]

    datasets_extraction_info = Parallel(n_jobs=-1)(
        delayed(traverse_project)(project_root, extension_to_language) for project_root in tqdm(repos_paths)
    )
    assert datasets_extraction_info is not None

    total_data = sum((info[0] for info in datasets_extraction_info), start=[])
    statuses = [info[1] for info in datasets_extraction_info]

    print(f"Number of errors: {sum(1 for status in statuses if status[0]==DownloadStatus.Error)}")

    with open(Config.output_path, "w") as f:
        json.dump(total_data, f, indent=2)


if __name__ == "__main__":
    main()
