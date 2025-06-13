from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import IterableDatasetDict
from tqdm.auto import tqdm


def calculate_accuracy(submission_tsv_path: Path | str) -> float:
    submission_df = pd.read_csv(submission_tsv_path, sep="\t").sort_values(by="id")

    assert "id" in submission_df.columns, (
        "提出された tsv には `id` 列が存在する必要があります"
    )
    assert "text" in submission_df.columns, (
        "提出された tsv には `text` 列が存在する必要があります"
    )

    test_data_dict = {"id": [], "text": []}

    test_data_count = 20000
    test_data_dict = {"id": [], "text": []}

    ds = load_dataset("taln-ls2n/kp20k", trust_remote_code=True, streaming=True)
    assert isinstance(ds, IterableDatasetDict)

    for data in tqdm(ds["test"], total=test_data_count, desc="Loading test data..."):
        test_data_dict["id"].append(data["id"])
        test_data_dict["text"].append(data["abstract"])

    test_df = pd.DataFrame(test_data_dict).sort_values(by="id")

    assert submission_df["id"].equals(test_df["id"]), (
        "テストデータセットと id が一致しません"
    )

    def calculate_accuracy_impl(hypothesis: str, reference: str) -> tuple[int, int]:
        assert len(hypothesis) == len(reference), (
            "prediction とラベルの長さが異なります"
        )
        correct_count = sum(c1 == c2 for c1, c2 in zip(hypothesis, reference))
        total_count = len(hypothesis)

        return total_count, correct_count

    # calculate accuracy
    total_count = 0
    correct_count = 0

    for hypothesis, reference in zip(submission_df["text"], test_df["text"]):
        total_count_iter, correct_count_iter = calculate_accuracy_impl(
            hypothesis, reference
        )
        total_count += total_count_iter
        correct_count += correct_count_iter

    return correct_count / total_count


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    print(f"Accuracy against test data is {calculate_accuracy(args.path):.3f}")
