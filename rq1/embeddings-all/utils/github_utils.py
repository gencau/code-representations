import os
import subprocess

class GitUtils:
    @staticmethod
    def saveFromGitHub(repo_root, owner, name, base_sha) -> str:
        output_dir = f"{repo_root}/{owner}_{name}_{base_sha}"

        # Don't recreate directory if it already exists
        if os.path.exists(output_dir):
            print(f"Repo snapshot already exists at {output_dir}")
            return output_dir
        
        # Save the repo snapshot to disk
        # initialize repo
        subprocess.run([
            "git", "init", output_dir
        ], check=True)

        # copy version of repo at sha. Here we do a shallow copy, not including full history but
        # just a snapshot in time
        subprocess.run([
            "git", "-C", output_dir, "remote", "add", "origin", f"https://github.com/{owner}/{name}.git"
        ], check=True)

        subprocess.run([
            "git", "-C", output_dir, "fetch", "--depth", "1", "origin", base_sha
        ], check=True)

        subprocess.run([
            "git", "-C", output_dir, "checkout", "FETCH_HEAD"
        ], check=True)
        print(f"Saved repo snapshot to {output_dir}")
        return output_dir