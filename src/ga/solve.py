import pandas as pd

from evaluation.utils import get_train_valid_tsv_path


class UnigramWithGASolver:
    def __calculate_hit_count(
        self, word_to_freq_dict: dict, char_to_token_mapping: dict
    ) -> int:
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

        hit_count = 0
        for sentence, tokenized_sentence in zip(
            train_df["text"], train_df["tokenized_text"]
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
