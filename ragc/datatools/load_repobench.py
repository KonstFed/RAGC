import argparse
import shutil
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

CUR_FILE_P = Path(__file__).parent.resolve()
LOAD_SCRIPT_PATH = CUR_FILE_P / "load_repo.bash"
N_PROCESSES = 4


def load_repo(data: tuple[str, Path, datetime]) -> tuple[str, bool, str, str, str]:
    repo_name, save_path = data
    folder = repo_name.split("/")[-1]
    save_path = save_path / folder

    if save_path.exists():
        return repo_name, False, "", "", f"Already exists {save_path}"

    result = subprocess.run(
        ["bash", LOAD_SCRIPT_PATH, repo_name, save_path], capture_output=True, text=True
    )
    commit_hash = result.stdout.strip().split("\n")[-1]
    return repo_name, result.returncode == 0, commit_hash, result.stdout, result.stderr


def multiprocess_load(repos2load: set[str], save_path: Path) -> pd.DataFrame:
    with Pool(processes=N_PROCESSES) as pool:
        results = tqdm(
            pool.imap_unordered(load_repo, [(r, save_path) for r in repos2load]),
            total=len(repos2load),
        )
        results = pd.DataFrame(
            results, columns=["repo", "status", "commit_hash", "stdout", "stder"]
        )
        results["reponame"] = results["repo"].apply(lambda p: p.split("/")[-1])

    return results


def load(save_path: Path, clean: bool = False) -> None:
    dataset = load_dataset("tianyang/repobench_python_v1.1")
    cross_file_rnd_ds = dataset["cross_file_random"].to_pandas()
    repos2load = set(cross_file_rnd_ds["repo_name"].unique())
    total_repobench_n_repo = len(repos2load)

    meta_p = save_path / "meta.csv"
    if meta_p.exists():
        meta = pd.read_csv(meta_p)
        repos2load = repos2load.difference(set(meta["repo"]))

    current_n_repo = len(repos2load)

    results = multiprocess_load(repos2load, save_path)
    loaded_results = results.loc[results["status"], ["repo", "reponame", "commit_hash"]]

    print(f"Loaded {len(loaded_results)}/{current_n_repo}")

    if meta_p.exists():
        meta = pd.read_csv(meta_p)
        meta = pd.concat([meta, loaded_results], axis=0)
    else:
        meta = loaded_results

    print(f"Total RepoBench loaded {len(meta)}/{total_repobench_n_repo}")

    meta.to_csv(meta_p, index=False)

    if not clean:
        return

    verified_folders = set(meta["reponame"])
    folder_iter = lambda: filter(
        lambda p: p.is_dir() and not p.stem.startswith("."), save_path.iterdir()
    )
    for folder in tqdm(list(folder_iter())):
        if folder.name in verified_folders:
            continue
        shutil.rmtree(folder)

    meta["check"] = False
    for folder in tqdm(list(folder_iter())):
        if folder.name in verified_folders:
            meta.loc[meta["reponame"] == folder.name, "check"] = True

    failed_repos = meta[~meta["check"]]
    assert len(failed_repos) == 0, (
        f"Sanity check failed, some verified files are incorrect in meta.csv for {len(failed_repos)} repos."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load repobench with repositories")
    parser.add_argument("output", type=Path)
    parser.add_argument("-c", "--clean", action="store_true")

    args = parser.parse_args()
    load(args.output, args.clean)
