# one_row_keyboard

## Installation

### Packages

```console
$ uv sync
```

### Dataset

```console
$ python preprocess/prepare_dataset.py
```

## Evaluation

1. テストデータセットの TSV を取得する
    ```python
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

    print(test_df.head())

    """
            id                                     tokenized_text
    0  44rSAMr  0 54431020 147943 849 45 0 67057 0 88 0 849 0 ...
    1  3MycQtg  0788 0798214 57454848 9427386048 94 5743829 97...
    2  4qxXN8m  0094822034 541443342783450974 20338380888 4294...
    3  3HdhbZQ  03 9788 505470 24 24388347 03 43970154 5472010...
    4  bUxKnm7  03 9788 748407270 0 342 9454 45 2030502907836 ...
    """
    ```
2. テストデータセットに対する推論結果を TSV として出力する
   (推論結果を `text` 列に格納する必要がある)
   ```python
    def inference(tokenized_text: str) -> str:
        # ......
        return pred_str

    test_df["text"] = test_df["tokenized_text"].apply(inference)
    test_df = test_df.drop("tokenized_text", axis="columns")

    print(test_df.head())
    # すべてを "e" と予測した場合
    """
                id                                               text
    0  44rSAMr  e eeeeeeee eeeeee eee ee e eeeee e ee e eee e ...
    1  3MycQtg  eeee eeeeeee eeeeeeee eeeeeeeeee ee eeeeeee ee...
    2  4qxXN8m  eeeeeeeeee eeeeeeeeeeeeeeeeee eeeeeeeeeee eeee...
    3  3HdhbZQ  ee eeee eeeeee ee eeeeeeee ee eeeeeeee eeeeeee...
    4  bUxKnm7  ee eeee eeeeeeeee e eee eeee ee eeeeeeeeeeeee ...
    """

    test_df.to_csv("submission.tsv", sep="\t")
   ```
3. 推論結果に対する accuracy を取得する
   ```console
   $ python -m evaluation.calculate_accuracy --path submission.tsv
   ```
