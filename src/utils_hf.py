"""HuggingFace dataset / wheel-cache helpers.

Centralizes the snapshot_download and upload calls that the notebook and
the local smoke test both need so the migration from Google Drive to the
HF dataset repo is one line in either context.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = "ruuudy/FinDrama"
DEFAULT_REPO_TYPE = "dataset"


def download_data(
    local_dir: str | os.PathLike = "./",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    train_zip: str = "data/train.tar.zip",
    val_zip: str = "data/validation.tar.zip",
    token: str | None = None,
) -> tuple[str, str]:
    """Download the FinDrama train and validation tar zips from HF.

    Pin a revision in calling code for reproducibility. Returns the absolute
    paths of the two downloaded files.
    """
    from huggingface_hub import snapshot_download

    target = Path(local_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        allow_patterns=[train_zip, val_zip],
        revision=revision,
        token=token,
        local_dir=str(target),
    )
    return str(target / train_zip), str(target / val_zip)


def upload_checkpoints(
    local_dir: str | os.PathLike,
    repo_id: str = DEFAULT_REPO_ID,
    path_in_repo: str = "checkpoints/lob",
    token: str | None = None,
    commit_message: str = "Upload pretraining checkpoints",
) -> None:
    """Upload a local checkpoint directory to the FinDrama HF dataset repo."""
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=str(Path(local_dir).resolve()),
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        path_in_repo=path_in_repo,
        token=token,
        commit_message=commit_message,
    )
