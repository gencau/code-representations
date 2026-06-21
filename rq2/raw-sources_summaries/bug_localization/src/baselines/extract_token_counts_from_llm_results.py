"""
This script extracts token counts from the metadata column in result files.
"""

import ast
import pandas as pd


def parse_metadata(value):
    """Parse metadata field and sum token counts across all entries."""
    import json

    if not isinstance(value, str):
        return {"input_tokens": None, "output_tokens": None}

    parsed = None
    # Try Python repr format first (handles True/False/None and single-quoted strings)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    # Try JSON format (handles double-quoted strings)
    if parsed is None:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {"input_tokens": None, "output_tokens": None}

    # Normalise to a list of dicts
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return {"input_tokens": None, "output_tokens": None}
    return {
        "input_tokens": sum(e.get("prompt_eval_count") or 0 for e in parsed if isinstance(e, dict)),
        "output_tokens": sum(e.get("eval_count") or 0 for e in parsed if isinstance(e, dict)),
    }


def extract_tokens(input_path: str, output_path: str = "output.csv", type="embeddings") -> pd.DataFrame:
    df = pd.read_csv(input_path)

    metadata_column = "rank_metadata"
        
    if metadata_column not in df.columns:
        # Try with response_metadata if metadata is not found
        metadata_column = "response_metadata"
        if "response_metadata" not in df.columns:
            if "rank_metadata" not in df.columns:
                raise ValueError(f"No '{metadata_column}' column found in the CSV.")
            else:
                metadata_column = "rank_metadata"

    token_data = df[metadata_column].apply(parse_metadata).apply(pd.Series)
    df["input_tokens"] = token_data["input_tokens"]
    df["output_tokens"] = token_data["output_tokens"]

    df.to_csv(output_path, index=False)
    print(f"Done. Saved to '{output_path}'")
    print(f"  Rows processed : {len(df)}")
    print(f"  Total input tokens : {df['input_tokens'].sum():,.0f}")
    print(f"  Total output tokens: {df['output_tokens'].sum():,.0f}")

    return df


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    type = sys.argv[3] if len(sys.argv) > 3 else "embeddings"
    extract_tokens(input_file, output_file, type)