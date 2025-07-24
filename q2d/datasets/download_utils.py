import os
import shutil
from enum import Enum
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
from git import RemoteProgress, Repo
from git.exc import GitError
from tqdm.auto import tqdm


class DownloadStatus(Enum):
    Done = 0
    Error = -1


class CloneProgress(RemoteProgress):
    OP_CODES = [
        "BEGIN",
        "CHECKING_OUT",
        "COMPRESSING",
        "COUNTING",
        "END",
        "FINDING_SOURCES",
        "RECEIVING",
        "RESOLVING",
        "WRITING",
    ]
    OP_CODE_MAP = {getattr(RemoteProgress, _op_code): _op_code for _op_code in OP_CODES}

    def __init__(self):
        super().__init__()
        self.pbar = None
        self.cur_op = -1

    @classmethod
    def get_curr_op(cls, op_code_masked: int) -> str:
        return cls.OP_CODE_MAP.get(op_code_masked, "?").title()

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ):
        # Remove BEGIN- and END-flag and get op name
        op_code_masked = op_code & CloneProgress.OP_MASK
        # init new tqdm for every op
        if self.cur_op != op_code_masked:
            self.pbar = tqdm()
            self.cur_op = op_code_masked
            self.pbar.desc = self.get_curr_op(op_code_masked)
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


def download_repo(
    git_url: str,
    repo_dir: str,
    force_reload: bool,
    git_rev: Optional[str] = None,
    retry: int = 2,
) -> None:
    if force_reload or not os.path.isdir(os.path.join(repo_dir, ".git")):
        if os.path.isdir(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        os.makedirs(repo_dir, exist_ok=True)

        try:
            cur_repo = Repo.clone_from(
                git_url,
                repo_dir,
                recurse_submodules=True,
                env={
                    "GIT_SSL_NO_VERIFY": "1",
                    "GIT_ASKPASS": "false",
                    "GIT_TERMINAL_PROMPT": "0",
                },
                progress=CloneProgress(),
                no_checkout=(git_rev is not None),
                **({} if (git_rev is not None) else {"depth": 1}),
            )

            if git_rev is not None:
                cur_repo.git.checkout(git_rev)

        except GitError as e:
            if retry > 0:
                sleep(5)
                return download_repo(
                    git_url,
                    repo_dir,
                    force_reload,
                    git_rev=git_rev,
                    retry=retry - 1,
                )
            raise e


def split_repo_url(repo_url: str) -> Tuple[str, Optional[str], str, str]:
    git_rev = repo_url.rsplit(":", 1)[1] if ":" in repo_url[6:] else None
    git_url = repo_url.rsplit(":", 1)[0] if ":" in repo_url[6:] else repo_url

    repo_maintainer = git_url.rsplit("/", 2)[1]
    repo_name = git_url.rsplit("/", 2)[2]
    return git_url, git_rev, repo_maintainer, repo_name


def download_repo_list(
    repos: List[str],
    main_repo_dir: str,
    force_reload: bool,
    retry: int = 2,
) -> List[Tuple[DownloadStatus, str]]:
    statuses: List[Tuple[DownloadStatus, str]] = []
    for repo_url in tqdm(repos):
        print(repo_url)
        git_url, git_rev, repo_maintainer, repo_name = split_repo_url(repo_url)

        try:
            download_repo(
                git_url=git_url,
                repo_dir=os.path.join(main_repo_dir, repo_maintainer, repo_name),
                force_reload=force_reload,
                git_rev=git_rev,
                retry=retry,
            )
            statuses.append((DownloadStatus.Done, "ok"))
        except GitError as e:
            statuses.append((DownloadStatus.Error, f"{str(type(e))}: {str(e)}"))

    return statuses


class Config:
    input_file = "./datasets/top_150.csv"
    main_repo_dir = "./datasets/diagrams-repos"
    force_reload = False
    retry = 2


def main():
    dataset = pd.read_csv(Config.input_file)
    repos = [data["repo"] for data in dataset.to_dict(orient="records")]
    download_status = download_repo_list(
        repos=repos, main_repo_dir=Config.main_repo_dir, force_reload=Config.force_reload, retry=Config.retry
    )

    print(f"Number of errors: {sum(1 for status in download_status if status[0]==DownloadStatus.Error)}")


if __name__ == "__main__":
    main()
