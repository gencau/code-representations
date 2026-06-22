from datasets import load_dataset
from pathlib import Path
from utils.dataset_parser import parse_lca, parse_swe_verified
from utils.record import Record

DATASETS = {
    "lca": dict(
        hf_id="JetBrains-Research/lca-bug-localization",
        arg_fn=lambda lang: (lang,),
        parser=parse_lca,
    ),
    "swe_verified": dict(
        hf_id="parquet",                               # local parquet loader
        arg_fn=lambda lang: {},                        # no config
        parser=parse_swe_verified,                     # NEEDS repo_root
    ),
}

class DatasetUtils:
    @staticmethod
    def get_records(ds_name: str,
                    ds_path: Path,
                    repo_root: Path | None,
                    language: str = "py",
                    k: int = 0,
                    stratified_sampling: bool = False) -> list[Record]:

        cfg = DATASETS[ds_name]

        # Load dataset
        if ds_name == "swe_verified":
            ds = load_dataset("parquet",
                              data_files={"test": str(ds_path)},
                              split="test")
        else:                                           
            ds = load_dataset(cfg["hf_id"],
                              *cfg["arg_fn"](language),
                              split="test")

        if stratified_sampling:
            shuffled = ds.shuffle(seed=42)
            # Select first k records
            if k:
                sampled_ds = shuffled.select(range(min(k, len(shuffled))))
            else:
                sampled_ds = shuffled
            
            # Parse the sampled records
            recs = [cfg["parser"](raw, repo_root=repo_root) for raw in sampled_ds]
            
            return recs
        
        # Fill records
        recs = []
        for i, raw in enumerate(ds):
            recs.append(cfg["parser"](raw, repo_root=repo_root))
            if k and i >= k-1:
                break
        return recs

