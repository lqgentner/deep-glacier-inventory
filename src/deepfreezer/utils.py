import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
"""Absolute base path of project. All paths are defined relative to this path."""


def download_file(
    url: str,
    dir: str | Path,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    """
    Download a file from a URL to a local directory, with optional authentication and progress bar.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dir : str or pathlib.Path
        The directory to save the downloaded file.
    username : str or None, optional
        Username for HTTP authentication (default is None).
    password : str or None, optional
        Password for HTTP authentication (default is None).

    Returns
    -------
    pathlib.Path
        The path of the downloaded file.
    """
    filename = url.split("/")[-1]
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    filepath = dir / filename

    with requests.Session() as session:
        if username and password:
            session.auth = (username, password)

        response = session.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=f"{_shorten_string(filename, 30): <30}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return filepath


def extract_zip(file: str | Path, delete_zip: bool = True) -> Path:
    """
    Extract a ZIP archive to its containing directory.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the ZIP file to extract.
    delete_zip : bool, optional
        If True, delete the ZIP file after extraction (default is True).

    Returns
    -------
    pathlib.Path
        The directory of the unzipped files
    """
    file = Path(file)
    unzip_dir = file.parent / file.stem
    zipfile.ZipFile(file).extractall(path=unzip_dir)
    if delete_zip:
        file.unlink()
    return unzip_dir


def _shorten_string(string: str, n: int) -> str:
    """Shorten a string to length n with ellipsis in the middle if needed."""
    if len(string) > n:
        n = n - len("...")
        n_1 = n // 2 + n % 2
        n_2 = n // 2
        string = string[:n_1] + "..." + string[-n_2:]
    return string
