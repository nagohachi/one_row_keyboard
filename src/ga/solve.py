from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

from evaluation.utils import get_train_valid_tsv_path


class UnigramWithGASolver:
    def __init__(self) -> None:
        train_path, _ = get_train_valid_tsv_path(
            char_to_token_mapping={}, unk_token_id=0
        )
        print("start reading train data...")
        train_df = pd.read_csv(train_path, sep="\t")
        print("done reading train data.")
        word_to_freq_dict = (
            train_df["text"].str.split().explode().value_counts().to_dict()
        )
        _, _, char_to_token_mapping, word_to_freq_dict = (
            self.calculate_conversion_dict()
        )
        hit_count = self.__calculate_hit_count(
            word_to_freq_dict=word_to_freq_dict,
            char_to_token_mapping=char_to_token_mapping,
        )
        print(f"Hit count: {hit_count}, {char_to_token_mapping=}")

    def __calculate_hit_count(
        self, word_to_freq_dict: dict, char_to_token_mapping: dict
    ) -> int:
        train_path, _ = get_train_valid_tsv_path(
            char_to_token_mapping=char_to_token_mapping, unk_token_id=0
        )
        train_df = pd.read_csv(train_path, sep="\t")

        conversion_dict = {}
        for sentence, tokenized_sentence in tqdm(
            zip(train_df["text"], train_df["tokenized_text"]),
            desc="Calculating conversion dict",
            total=len(train_df),
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                if (
                    tokenized_word in conversion_dict
                    and conversion_dict[tokenized_word] != word
                ):
                    if (
                        word_to_freq_dict[word]
                        > word_to_freq_dict[conversion_dict[tokenized_word]]
                    ):
                        conversion_dict[tokenized_word] = word
                conversion_dict[tokenized_word] = word

        hit_count = 0
        for sentence, tokenized_sentence in tqdm(
            zip(train_df["text"], train_df["tokenized_text"]),
            desc="Calculating hit count",
            total=len(train_df),
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                if (
                    word_to_freq_dict[word]
                    < word_to_freq_dict[conversion_dict[tokenized_word]]
                    and tokenized_word in conversion_dict
                ):
                    hit_count += 1

        return hit_count

    def calculate_conversion_dict(self) -> tuple[dict, dict, dict, dict]:
        train_path_dummy, _ = get_train_valid_tsv_path(
            char_to_token_mapping={}, unk_token_id=0
        )
        train_df = pd.read_csv(train_path_dummy, sep="\t")

        char_to_freq_dict = defaultdict(lambda: 0)
        word_to_freq_dict = defaultdict(lambda: 0)

        for sentence in tqdm(train_df["text"], desc="Calculating frequency"):
            if not isinstance(sentence, str):
                continue
            for word in sentence.split():
                if not word.strip():
                    continue
                word_to_freq_dict[word] += 1

                for char in word:
                    if not char.strip():
                        continue
                    char_to_freq_dict[char] += 1

        charset_with_freq_sorted = sorted(
            char_to_freq_dict.items(), key=lambda x: x[1], reverse=True
        )

        char_to_token_mapping = {}
        token_to_most_frequent_char_mapping = {}

        for i, (char, _) in enumerate(charset_with_freq_sorted):
            if char == " ":
                continue
            # Ensure high-frequency characters do not share the same token
            token = i % 10
            char_to_token_mapping[char] = token
            if str(token) not in token_to_most_frequent_char_mapping:
                token_to_most_frequent_char_mapping[str(token)] = char

        train_path, _ = get_train_valid_tsv_path(
            char_to_token_mapping=char_to_token_mapping, unk_token_id=0
        )
        train_df = pd.read_csv(train_path, sep="\t")

        conversion_dict = {}
        for sentence, tokenized_sentence in zip(
            train_df["text"], train_df["tokenized_text"]
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                if (
                    tokenized_word in conversion_dict
                    and conversion_dict[tokenized_word] != word
                ):
                    if (
                        word_to_freq_dict[word]
                        > word_to_freq_dict[conversion_dict[tokenized_word]]
                    ):
                        conversion_dict[tokenized_word] = word
                conversion_dict[tokenized_word] = word

        return (
            conversion_dict,
            token_to_most_frequent_char_mapping,
            char_to_token_mapping,
            word_to_freq_dict,
        )


if __name__ == "__main__":
    solver = UnigramWithGASolver()
