import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SWITCH_SCRIPT_P = Path(__file__).parent.resolve() / "switch_commit.bash"
SWITCH_SCRIPT_LATEST_P = Path(__file__).parent.resolve() / "switch_to_latest.bash"
N_PROCESSES = 8


def switch_single_repo(repo_p: Path, cutoff_date: datetime | None) -> tuple[str, bool, str]:    
    if cutoff_date is None:
        result = subprocess.run(["bash", SWITCH_SCRIPT_LATEST_P, str(repo_p)], capture_output=True, text=True)
    else:
        result = subprocess.run(["bash", SWITCH_SCRIPT_P, str(repo_p), cutoff_date], capture_output=True, text=True)
    
    commit_hash = result.stdout.strip().split("\n")[-1]
    return repo_p.name, result.returncode == 0, commit_hash

def switch(data_p: Path, cutoff_date: datetime | None) -> None:
    folders = list(filter(lambda p: p.is_dir() and not p.stem.startswith("."), data_p.iterdir()))
    results = []
    bar = tqdm(folders)
    for f in bar:
        bar.set_description(f.name)
        results.append(switch_single_repo(f, cutoff_date))

    results = pd.DataFrame(results, columns = ["reponame", "status", "commit_hash"])
    print("Sucessfully loaded ", results["status"].mean())
    results = results[results["status"]]

    meta = pd.read_csv(data_p / "meta.csv")

    meta = meta.merge(results[['reponame', 'commit_hash']], on='reponame', how='left', suffixes=('', '_new'))
    meta['commit_hash'] = meta['commit_hash_new'].combine_first(meta['commit_hash'])
    meta.drop(columns=['commit_hash_new'], inplace=True)
    
    meta.to_csv(data_p / "meta.csv", index=False)

    print("Sucessfully loaded ", results["status"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('-d', '--date', required=False)

    args = parser.parse_args()
    switch(args.input, args.date)   
