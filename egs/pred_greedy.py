"""one_row_keyboard/ 内で、`python -m egs.pred_greedy`"""

from pathlib import Path

import pandas as pd

from evaluation.calculate_accuracy import calculate_accuracy
from evaluation.utils import get_test_tsv_path

char_to_token_mapping = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 0,
    "l": 1,
    "m": 2,
    "n": 3,
    "o": 4,
    "p": 5,
    "q": 6,
    "r": 7,
    "s": 8,
    "t": 9,
    "u": 0,
    "v": 1,
    "w": 2,
    "x": 3,
    "y": 4,
    "z": 5,
}

# char_to_token_mapping の value は 0, 1, ..., 9 のいずれかである必要がある
# unk_token_id は char_to_token_mapping の key にない文字がテストデータに存在したときに与えるトークンで、
# 0, 1, ..., 9 のいずれかである必要がある
test_tsv_path = get_test_tsv_path(
    char_to_token_mapping=char_to_token_mapping, unk_token_id=0
)

test_df = pd.read_csv(test_tsv_path, sep="\t")
token_to_char_mapping = dict()

for char, token in char_to_token_mapping.items():
    if str(token) in token_to_char_mapping:
        continue
    token_to_char_mapping[str(token)] = char


def inference(tokenized_text: str) -> str:
    """char_to_token_mapping に含まれている最初の文字で復元する"""
    result = ""
    for token in tokenized_text:
        if token == " ":
            result += " "
        else:
            result += token_to_char_mapping[token]

    return result


test_df["text"] = test_df["tokenized_text"].apply(inference)
test_df = test_df.drop("tokenized_text", axis="columns")
test_df.to_csv("submission.tsv", sep="\t")

print(calculate_accuracy("submission.tsv"))  # 0.514

Path("submission.tsv").unlink()
