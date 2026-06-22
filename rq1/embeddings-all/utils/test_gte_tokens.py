from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)

def gte_tokens(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))

def estimate_tokens_for_2048_char_windows(text: str, window_chars=2048, overlap_chars=60):
    step = window_chars - overlap_chars
    counts = []
    for start in range(0, len(text), step):
        chunk = text[start:start+window_chars]
        if not chunk:
            break
        counts.append(gte_tokens(chunk))
        if start + window_chars >= len(text):
            break
    return counts

# Example usage: for each file content `code`
# counts = estimate_tokens_for_2048_char_windows(code)
# print(np.median(counts), np.percentile(counts, [25, 75, 90]))
if __name__ == "__main__":
    # Test with a sample string
    sample_text = "def example_function(param1, param2):\n    return param1 + param2\n" * 100
    token_count = gte_tokens(sample_text)
    print(f"Token count for sample text: {token_count}")

    # Test the estimation function
    counts = estimate_tokens_for_2048_char_windows(sample_text)
    print(f"Estimated token counts for 2048-char windows: {counts}")
    print(f"Median: {np.median(counts)}, 25th percentile: {np.percentile(counts, 25)}, 75th percentile: {np.percentile(counts, 75)}, 90th percentile: {np.percentile(counts, 90)}")