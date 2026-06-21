import os
import subprocess
from datasets import load_dataset

""" Gets all repos in the test split (50) for the bug
    localization task.
    Note that the script already does it, this is just required
    if you don't want to run the whole script.
"""
splits = ['test']
configurations = ['java','py','kt']

for configuration in configurations:
    dataset_java = load_dataset("JetBrains-Research/lca-bug-localization", configuration)

    for split in splits:
        output_dir = f"../cloned_repos_{split}/{configuration}"
        os.makedirs(output_dir, exist_ok=True)

        dataset = dataset_java[split]

        for row in dataset:
            owner = row['repo_owner']
            name = row['repo_name']
            base_sha = row['base_sha']
            #print(owner, name, base_sha)

            repo_path = os.path.join(output_dir, f"{owner}_{name}_{base_sha}")

            if os.path.exists(repo_path):
                print(f"Skipping {repo_path}, already exists!")
                continue

            # initialize repo
            subprocess.run([
                "git", "init", repo_path
            ], check=True)

            # copy version of repo at sha. Here we do a shallow copy, not including full history but
            # just a snapshot in time
            subprocess.run([
                "git", "-C", repo_path, "remote", "add", "origin", f"https://github.com/{owner}/{name}.git"
            ], check=True)

            subprocess.run([
                "git", "-C", repo_path, "fetch", "--depth", "1", "origin", base_sha
            ], check=True)

            subprocess.run([
                "git", "-C", repo_path, "checkout", "FETCH_HEAD"
            ], check=True)


print(f"All Done!! See repos in {output_dir}") 