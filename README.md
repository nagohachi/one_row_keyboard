# one_row_keyboard

## About

- かな漢字変換等をテーマにした、言語モデルの作成を行う。
- あらかじめ、データセット内の各文字を「0, 1, ..., 9」に変換するマッピングを決める。
- 「0, 1, ..., 9」およびスペースからなる文が与えられるので、自分で決めたマッピング、および文脈の情報を使い、元の文を復元する。
- 以下では、**元の文を「文字列」と呼び、「0, 1, ..., 9」およびスペースからなる文を「トークン列」と呼ぶ**。

### Examples

スクリプトの例が [egs/](egs/) にいくつか存在するので参照のこと。

以下では、英語が対象言語であるとする。
マッピングを以下のように定める: 
```
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
```
このとき、`a keyboard` という文字列は `0 04414073` のように変換される。

テスト時は、マッピングを与えると、正解の文字列をマッピングにより変換したトークン列が得られる (後の Evaluation セクションを参照のこと)。<br>テストデータに対しては、このトークン列を入力とし、元の文字列を復元する。<br>
今回の例であると、`0 04414073` が与えられるので `a keyboard` を復元する。
このとき、マッピングを考えるとトークン `0` は `a`, `k`, `u` のいずれにも復元可能である。<br>しかしながら、英語の特性を考えた場合、(ある程度フォーマルな文であれば) 1文字語として考えられるのは `a` のみである。

このような amgibuous な入力に対して、（言語的特性を踏まえて）出力を予想するモデルを構築する。<br>
また、言語的特性を学習するための学習データセット・検証データセットも用意している。Installation > Dataset セクションを参照のこと。

## Installation

### Packages

```console
$ uv sync
```

## Evaluation

1. テストデータセットの TSV を取得する
    ```python
    from evaluation.utils import get_test_tsv_path, get_train_valid_tsv_path

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
    train_tsv_path, valid_tsv_path = get_train_valid_tsv_path(
        char_to_token_mapping=char_to_token_mapping, unk_token_id=0
    )

    test_tsv_path = get_test_tsv_path(
        char_to_token_mapping=char_to_token_mapping, unk_token_id=0
    )

    # train_df = pd.read_csv(train_tsv_path, sep="\t")
    # valid_df = pd.read_csv(valid_tsv_path, sep="\t")
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
   $ # Accuracy against test data is 0.249
   ```
