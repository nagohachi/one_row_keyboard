from functools import partial
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from constants.paths import INPUT_DIR

token_id_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def tokenize(
    text: str, char_to_token_mapping: dict[str, int], unk_token_id: int
) -> str:
    """text -> tokenized text"""
    tokenized_text = ""
    for c in text:
        if c == " ":
            tokenized_text += " "
        elif c not in char_to_token_mapping:
            tokenized_text += str(unk_token_id)
        else:
            tokenized_text += str(char_to_token_mapping[c])
    return tokenized_text


def batch_tokenize(
    examples: dict, char_to_token_mapping: dict[str, int], unk_token_id: int
) -> dict:
    text_list = examples["abstract"]

    return {
        "text": text_list,
        "tokenized_text": [
            tokenize(text, char_to_token_mapping, unk_token_id) for text in text_list
        ],
    }


def get_train_valid_tsv_path(
    char_to_token_mapping: dict[str, int],
    unk_token_id: int,
) -> tuple[Path, Path]:
    assert " " not in char_to_token_mapping, (
        "半角スペースはトークン化する必要はありません"
    )
    assert set(char_to_token_mapping.values()) <= token_id_set, (
        f"char_to_token_mapping は str -> {token_id_set} である必要があります"
    )
    assert unk_token_id in token_id_set, (
        f"unk_token_id は {token_id_set} の要素である必要があります"
    )

    ds = load_dataset("taln-ls2n/kp20k", trust_remote_code=True)
    assert isinstance(ds, DatasetDict)

    train_dataset = ds["train"].map(
        partial(
            batch_tokenize,
            char_to_token_mapping=char_to_token_mapping,
            unk_token_id=unk_token_id,
        ),
        remove_columns=["title", "abstract", "keyphrases", "prmu"],
        batched=True,
    )

    valid_dataset = ds["validation"].map(
        partial(
            batch_tokenize,
            char_to_token_mapping=char_to_token_mapping,
            unk_token_id=unk_token_id,
        ),
        remove_columns=["title", "abstract", "keyphrases", "prmu"],
        batched=True,
    )

    train_tsv_path = INPUT_DIR / "train.tsv"
    valid_tsv_path = INPUT_DIR / "valid.tsv"

    train_dataset.to_pandas().to_csv(train_tsv_path, sep="\t", index=False)  # type: ignore
    valid_dataset.to_pandas().to_csv(valid_tsv_path, sep="\t", index=False)  # type: ignore

    return train_tsv_path, valid_tsv_path


def get_test_tsv_path(
    char_to_token_mapping: dict[str, int],
    unk_token_id: int,
) -> Path:
    assert " " not in char_to_token_mapping, (
        "半角スペースはトークン化する必要はありません"
    )
    assert set(char_to_token_mapping.values()) <= token_id_set, (
        f"char_to_token_mapping は str -> {token_id_set} である必要があります"
    )
    assert unk_token_id in token_id_set, (
        f"unk_token_id は {token_id_set} の要素である必要があります"
    )

    ds = load_dataset("taln-ls2n/kp20k", trust_remote_code=True)
    assert isinstance(ds, DatasetDict)

    test_dataset = ds["test"].map(
        partial(
            batch_tokenize,
            char_to_token_mapping=char_to_token_mapping,
            unk_token_id=unk_token_id,
        ),
        remove_columns=["title", "abstract", "keyphrases", "prmu"],
        batched=True,
    )

    test_tsv_path = INPUT_DIR / "test.tsv"

    test_dataset.to_pandas().drop(columns="text").to_csv(  # type: ignore
        test_tsv_path, sep="\t", index=False
    )

    return test_tsv_path


if __name__ == "__main__":
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

    train_tsv_path, valid_tsv_path = get_train_valid_tsv_path(
        char_to_token_mapping=char_to_token_mapping, unk_token_id=0
    )

    test_tsv_path = get_test_tsv_path(
        char_to_token_mapping=char_to_token_mapping, unk_token_id=0
    )

    test_df = pd.read_csv(test_tsv_path, sep="\t")

    print(test_df.head())

    """
            id                                     tokenized_text
    0  44rSAMr  0 54431020 147943 849 45 0 67057 0 88 0 849 0 ...
    1  3MycQtg  0788 0798214 57454848 9427386048 94 5743829 97...
    2  4qxXN8m  0094822034 541443342783450974 20338380888 4294...
    3  3HdhbZQ  03 9788 505470 24 24388347 03 43970154 5472010...
    4  bUxKnm7  03 9788 748407270 0 342 9454 45 2030502907836 ...
    """
