"""Model loading utilities."""

import importlib.metadata
import os
from pathlib import Path

import requests


class ModelDownloadError(Exception):
    """Exception raised when model downloading fails."""

    pass


EXPECTED_MODEL_FILENAMES = (
    "anchor_det_model_512px.keras",
    "root_det_model_512px.keras",
)


def get_models_dir(version: str | None = None, force_download: bool = False) -> Path:
    """Return path to a directory containing all required .keras model files.

    Args:
        version: Package version to fetch. Defaults to importlib.metadata.version("npec_cv").
        force_download: Re-download even if cache is populated.

    Raises:
        ModelDownloadError: If GITHUB_TOKEN is missing, the release doesn't exist,
            or any asset fails to download.
    """
    if version is None:
        try:
            version = importlib.metadata.version("npec_cv")
        except importlib.metadata.PackageNotFoundError:
            version = "0.2.0"  # fallback if not installed properly

    default_cache = Path.home() / ".cache" / "npec_cv" / "models"
    cache_dir = Path(os.getenv("NPEC_MODELS_CACHE_DIR", default_cache)) / f"v{version}"

    # If all models exist, return the cache_dir
    if not force_download and cache_dir.exists():
        if all((cache_dir / filename).exists() for filename in EXPECTED_MODEL_FILENAMES):
            return cache_dir

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ModelDownloadError(
            "GITHUB_TOKEN not set. Generate a Personal Access Token with 'repo' scope at "
            "https://github.com/settings/tokens, then export GITHUB_TOKEN=<your-token>."
        )

    print(f"Downloading models for npec_cv v{version}...")
    headers = {"Authorization": f"Bearer {github_token}"}

    # Try getting the release info from GitHub API
    # Since this is a private repo, we use the Github API to get the release by tag.
    # Wait, the prompt says "private repo's GitHub Releases". The spec says:
    # "https://api.github.com/repos/<owner>/<repo>/releases/tags/v<version>"
    repo = "RazvanNica242715/cv5-clone"  # Path to repo clone - replace with actual owner/repo later
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/v{version}"

    resp = requests.get(release_url, headers=headers)
    if resp.status_code == 404:
        raise ModelDownloadError(f"Release v{version} not found in {repo}.")
    resp.raise_for_status()
    release_data = resp.json()

    assets = release_data.get("assets", [])
    asset_by_name = {asset["name"]: asset["url"] for asset in assets}

    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in EXPECTED_MODEL_FILENAMES:
        if filename not in asset_by_name:
            raise ModelDownloadError(f"Asset {filename} missing from release v{version}.")

        asset_url = asset_by_name[filename]
        print(f"Downloading {filename}...")

        # Download as stream
        dl_resp = requests.get(
            asset_url,
            headers={"Authorization": f"Bearer {github_token}", "Accept": "application/octet-stream"},
            stream=True,
        )
        dl_resp.raise_for_status()

        tmp_path = cache_dir / f"{filename}.tmp"
        with open(tmp_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                f.write(chunk)

        tmp_path.rename(cache_dir / filename)

    return cache_dir
