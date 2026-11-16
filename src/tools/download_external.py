"""Download the open data about OTHER matches used by the test bench, into data/external/.

    python src/tools/download_external.py                 # all sources
    python src/tools/download_external.py idsse metrica   # only these

Sources, all public, all about other games (RULES.md section 2), ~4.7 GB together:

  * metrica-sample  3 anonymous matches, 25 fps tracking + human event data, ~180 MB
                    github.com/metrica-sports/sample-data. no licence, credit Metrica Sports
  * idsse           7 Bundesliga 2022/23 matches, 25 fps TRACAB tracking with per-frame
                    BallStatus and BallPossession, + events, ~2.6 GB
                    huggingface.co/datasets/pysport/idsse-data (mirror of figshare 28196177)
                    CC-BY 4.0, credit the DFL, cite Bassek et al. 2025, doi.org/10.1038/s41597-025-04505-y
  * skillcorner     20 A-League 2024/25 matches, 10 fps broadcast tracking with per-frame
                    possession group and ball is_detected, + phases of play and events, ~1.9 GB
                    github.com/SkillCorner/opendata. MIT licence, credit SkillCorner

Behaviour:
  * a file is skipped when its local size equals the server's Content-Length, so a re-run only
    fetches what is missing or broken
  * files are written to <name>.part and renamed when complete
  * github.com/<repo>/raw/ serves both normal and git-lfs files
  * SkillCorner aggregates, body pose and tutorial assets are not downloaded
"""

import json
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Params
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")
USER_AGENT = "octo-mtah-download"
CHUNK = 1 << 20


def open_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def github_files(repo, branch, keep):
    """Paths in a GitHub repo that start with one of the `keep` prefixes."""
    with open_url("https://api.github.com/repos/{0}/git/trees/{1}?recursive=1".format(repo, branch)) as response:
        tree = json.load(response)["tree"]
    return [entry["path"] for entry in tree if entry["type"] == "blob" and entry["path"].startswith(keep)]


def huggingface_files(dataset):
    """Top-level file paths of a Hugging Face dataset, without the dotfiles."""
    with open_url("https://huggingface.co/api/datasets/{0}/tree/main".format(dataset)) as response:
        tree = json.load(response)
    return [entry["path"] for entry in tree if entry["type"] == "file" and not entry["path"].startswith(".")]


def fetch(url, target):
    """Download `url` to `target` unless an identical-size file is already there. True if downloaded."""
    with open_url(url) as response:
        size = int(response.headers.get("Content-Length", -1))
        if os.path.exists(target) and os.path.getsize(target) == size:
            return False
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target + ".part", "wb") as out:
            shutil.copyfileobj(response, out, CHUNK)
    os.replace(target + ".part", target)
    return True


def main():
    sources = {
        "metrica-sample": [
            ("https://github.com/metrica-sports/sample-data/raw/master/" + path, path)
            for path in github_files(repo="metrica-sports/sample-data", branch="master",
                                     keep=("data/", "documentation/", "README.md"))],
        "idsse": [
            ("https://huggingface.co/datasets/pysport/idsse-data/resolve/main/" + path, path)
            for path in huggingface_files(dataset="pysport/idsse-data")],
        "skillcorner": [
            ("https://github.com/SkillCorner/opendata/raw/master/" + path, path)
            for path in github_files(repo="SkillCorner/opendata", branch="master",
                                     keep=("data/matches/", "data/matches.json", "README.md", "LICENSE"))],
    }
    chosen = sys.argv[1:] if len(sys.argv) > 1 else list(sources)
    for name in chosen:
        files = sources[name]
        print("{0}: {1} files".format(name, len(files)), flush=True)
        for index, (url, path) in enumerate(files):
            target = os.path.join(EXTERNAL_DIR, name, *path.split("/"))
            downloaded = fetch(url=url, target=target)
            print("  [{0}/{1}] {2} {3}".format(index + 1, len(files), "got " if downloaded else "have", path), flush=True)
    print("done")


if __name__ == "__main__":
    main()
