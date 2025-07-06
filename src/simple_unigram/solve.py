import re
from collections import defaultdict

import pandas as pd

from evaluation.calculate_accuracy import calculate_accuracy
from evaluation.utils import get_test_tsv_path, get_train_valid_tsv_path


class SimpleUnigramSolver:
    def solve(self) -> None:
        conversion_dict, token_to_most_frequent_char_dict, char_to_token_mapping = (
            self.calculate_conversion_dict()
        )
        self.__solve_impl(
            conversion_dict=conversion_dict,
            token_to_most_frequent_char_dict=token_to_most_frequent_char_dict,
            char_to_token_mapping=char_to_token_mapping,
        )

    def __solve_impl(
        self,
        conversion_dict: dict,
        token_to_most_frequent_char_dict: dict,
        char_to_token_mapping: dict,
    ) -> None:
        def replacer(match: re.Match) -> str:
            tokenized_word = match.group(0)
            # If the tokenized word is registered, convert it to the most probable word
            if tokenized_word in conversion_dict:
                return conversion_dict[tokenized_word]
            # If not registered, convert it to the most probable character sequence
            return "".join(
                [token_to_most_frequent_char_dict[token] for token in tokenized_word]
            )

        test_path = get_test_tsv_path(
            char_to_token_mapping=char_to_token_mapping, unk_token_id=0
        )
        test_df = pd.read_csv(test_path, sep="\t")
        test_df["text"] = test_df["tokenized_text"].apply(
            lambda x: re.sub(r"\d+", replacer, x)
        )
        test_df.to_csv("sub.tsv", sep="\t", index=False, columns=["id", "text"])
        print(f"Accuracy: {calculate_accuracy('sub.tsv')}")

    def calculate_conversion_dict(self) -> tuple[dict, dict, dict]:
        train_path_dummy, _ = get_train_valid_tsv_path(
            char_to_token_mapping={}, unk_token_id=0
        )
        train_df = pd.read_csv(train_path_dummy, sep="\t")

        char_to_freq_dict = defaultdict(lambda: 0)
        word_to_freq_dict = defaultdict(lambda: 0)

        for sentence in train_df["text"]:
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
        )


if __name__ == "__main__":
    solver = SimpleUnigramSolver()
    solver.solve()
