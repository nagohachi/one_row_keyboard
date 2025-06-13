import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import IterableDatasetDict
from tqdm.auto import tqdm

from constants.paths import INPUT_DIR


def prepare_dataset() -> None:
    ds = load_dataset("taln-ls2n/kp20k", trust_remote_code=True, streaming=True)

    assert isinstance(ds, IterableDatasetDict)

    train_data_dict = {"id": [], "text": []}
    valid_data_dict = {"id": [], "text": []}

    train_data_count = 530809
    valid_data_count = 20000

    for data in tqdm(ds["train"], total=train_data_count):
        train_data_dict["id"].append(data["id"])
        train_data_dict["text"].append(data["abstract"])

    for data in tqdm(ds["validation"], total=valid_data_count):
        valid_data_dict["id"].append(data["id"])
        valid_data_dict["text"].append(data["abstract"])

    pd.DataFrame(train_data_dict).to_csv(INPUT_DIR / "train.tsv", sep="\t", index=False)
    pd.DataFrame(valid_data_dict).to_csv(INPUT_DIR / "valid.tsv", sep="\t", index=False)


if __name__ == "__main__":
    prepare_dataset()
