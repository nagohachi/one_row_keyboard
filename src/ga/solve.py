import re
import random
from collections import defaultdict
from typing import List
import json
import traceback

import pandas as pd
from tqdm.auto import tqdm

from evaluation.calculate_accuracy import calculate_accuracy
from evaluation.utils import get_test_tsv_path, get_train_valid_tsv_path


class UnigramWithGASolver:
    """遺伝的アルゴリズムを使用したユニグラムソルバー"""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
    ) -> None:
        """初期化処理"""
        print("=== Initializing UnigramWithGASolver ===")
        print(
            f"Parameters: population_size={population_size}, generations={generations}, mutation_rate={mutation_rate}"
        )

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.train_df = self._load_train_data()
        print(f"Loaded training data: {len(self.train_df)} samples")

        self.word_to_freq_dict = self._calculate_word_frequencies()
        print(
            f"Calculated word frequencies: {len(self.word_to_freq_dict)} unique words"
        )

        # 初期のchar_to_token_mapping（キーは固定、値は最適化対象）
        self.initial_char_to_token_mapping = {
            "e": 0,
            "t": 1,
            "i": 2,
            "a": 3,
            "o": 4,
            "n": 5,
            "s": 6,
            "r": 7,
            "l": 8,
            "c": 9,
            "h": 0,
            "d": 1,
            "m": 2,
            "p": 3,
            "u": 4,
            "f": 5,
            "g": 6,
            "y": 7,
            "b": 8,
            "v": 9,
            "w": 0,
            ".": 1,
            ",": 2,
            "-": 3,
            "k": 4,
            "T": 5,
            "x": 6,
            ")": 7,
            "(": 8,
            "A": 9,
            "I": 0,
            "S": 1,
            "q": 2,
            "z": 3,
            "C": 4,
            "M": 5,
            "P": 6,
            "W": 7,
            "D": 8,
            "E": 9,
            "0": 0,
            "F": 1,
            "R": 2,
            "2": 3,
            "1": 4,
            "O": 5,
            "L": 6,
            "B": 7,
            "N": 8,
            "j": 9,
            "G": 0,
            "H": 1,
            "V": 2,
            "3": 3,
            "/": 4,
            "U": 5,
            "9": 6,
            "5": 7,
            ":": 8,
            "'": 9,
            "4": 0,
            "?": 1,
            "6": 2,
            "8": 3,
            "K": 4,
            "7": 5,
            ";": 6,
            "%": 7,
            "Q": 8,
            '"': 9,
            "J": 0,
            "\\": 1,
            "X": 2,
            "+": 3,
            "=": 4,
            "[": 5,
            "]": 6,
            "Y": 7,
            "Z": 8,
            "&": 9,
            "{": 0,
            "}": 1,
            "_": 2,
            ">": 3,
            "<": 4,
            "|": 5,
            "*": 6,
            "^": 7,
            "$": 8,
            "!": 9,
            "@": 0,
            "#": 1,
            "`": 2,
            "~": 3,
        }
        print(
            f"Initial char_to_token_mapping: {len(self.initial_char_to_token_mapping)} characters"
        )

        # 最適化されたマッピング
        print("\n=== Starting Genetic Algorithm Optimization ===")
        self.char_to_token_mapping = self._optimize_mapping()

        print("\n=== Creating Final Conversion Dictionary ===")
        self.conversion_dict = self._create_conversion_dict()
        print(
            f"Created conversion dictionary: {len(self.conversion_dict)} tokenized words"
        )

        self.token_to_most_frequent_char_mapping = self._create_token_to_char_mapping()
        print(
            f"Created token_to_char mapping: {len(self.token_to_most_frequent_char_mapping)} tokens"
        )

        print("\n=== Calculating Final Hit Count ===")
        hit_count = self._calculate_hit_count()
        print(f"Final optimized hit count: {hit_count}")
        print(f"Final char_to_token_mapping: {self.char_to_token_mapping}")

    def _optimize_mapping(self) -> dict[str, int]:
        """遺伝的アルゴリズムでマッピングを最適化する"""
        print("Starting genetic algorithm optimization...")
        print(
            f"Population size: {self.population_size}, Generations: {self.generations}, Mutation rate: {self.mutation_rate}"
        )

        # 初期集団を生成
        print("Generating initial population...")
        population = self._generate_initial_population()
        print(f"Initial population size: {len(population)}")

        best_mapping = None
        best_fitness = float("inf")

        for generation in tqdm(range(self.generations), desc="Genetic Algorithm"):
            print(f"\n--- Generation {generation + 1} ---")

            # 適応度を計算
            print("Calculating fitness for all individuals...")
            fitness_scores = []
            for i, individual in enumerate(
                tqdm(population, desc=f"Fitness (Gen {generation + 1})")
            ):
                fitness = self._calculate_fitness(individual)
                fitness_scores.append(fitness)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_mapping = individual.copy()
                    print(f"    -> New best! Fitness improved to {best_fitness}")

            # 各世代ごとにfitnessが良い個体を上位3つjson保存
            top3_indices = sorted(
                range(len(fitness_scores)), key=lambda i: fitness_scores[i]
            )[:3]
            for rank, idx in enumerate(top3_indices):
                best_individual = population[idx]
                fitness = fitness_scores[idx]
                with open(
                    f"best_mapping_gen{generation + 1}_rank{rank + 1}_fitness{fitness}.json",
                    "w",
                ) as f:
                    json.dump(best_individual, f, ensure_ascii=False, indent=2)

            print(f"Generation {generation + 1} summary:")
            print(f"  Best fitness: {best_fitness}")
            print(f"  Average fitness: {sum(fitness_scores) / len(fitness_scores):.2f}")
            print(f"  Worst fitness: {max(fitness_scores)}")

            # 新しい集団を生成
            print("Creating new population...")
            new_population = []

            # エリート選択（最良の個体を保持）
            elite_size = max(1, self.population_size // 10)
            elite_indices = sorted(
                range(len(fitness_scores)), key=lambda i: fitness_scores[i]
            )[:elite_size]
            print(f"  Elite selection: keeping {elite_size} best individuals")
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # 交叉と突然変異で残りの個体を生成
            print(
                f"  Creating {self.population_size - len(new_population)} new individuals through crossover and mutation..."
            )
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            print(f"New population size: {len(population)}")

        print(f"\n=== Optimization completed ===")
        print(f"Best fitness achieved: {best_fitness}")
        print(f"Best mapping: {best_mapping}")
        return best_mapping or self.initial_char_to_token_mapping

    def _generate_initial_population(self) -> List[dict[str, int]]:
        """初期集団を生成する"""
        population = []

        # 初期マッピングを1つ目に追加
        print("  Adding initial mapping as first individual")
        population.append(self.initial_char_to_token_mapping.copy())

        # 初期解を突然変異させた個体を2つ目に追加
        print("  Adding mutated initial mapping as second individual")
        mutated = self._mutate(self.initial_char_to_token_mapping.copy())
        population.append(mutated)

        # 残りをランダムに生成
        print(f"  Generating {self.population_size - 2} random individuals...")
        for i in range(self.population_size - 2):
            individual = {}
            for char in self.initial_char_to_token_mapping.keys():
                individual[char] = random.randint(0, 9)
            population.append(individual)
            if (i + 1) % 10 == 0:
                print(f"    Generated {i + 1} random individuals")

        return population

    def _calculate_fitness(self, char_to_token_mapping: dict[str, int]) -> int:
        """適応度（hit count）を計算する"""
        try:
            # 指定されたマッピングで変換辞書を作成
            conversion_dict = self._create_conversion_dict_with_mapping(
                char_to_token_mapping
            )

            # hit countを計算
            hit_count = self._calculate_hit_count_with_dict_and_mapping(
                conversion_dict, char_to_token_mapping
            )

            return hit_count
        except Exception as e:
            traceback.print_exc()
            print(f"Error in fitness calculation: {e}")
            return 1000000

    def _create_conversion_dict_with_mapping(
        self, char_to_token_mapping: dict[str, int]
    ) -> dict[str, str]:
        """指定されたマッピングで変換辞書を作成する"""
        # トークン化されたデータを読み込む
        train_df, _ = get_train_valid_tsv_path(
            char_to_token_mapping=char_to_token_mapping,
            unk_token_id=0,
            use_existing_tsv=True,
            return_dataframe=True,
        )
        tokenized_df = train_df

        conversion_dict = {}
        for sentence, tokenized_sentence in zip(
            tokenized_df["text"], tokenized_df["tokenized_text"]
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                self._update_conversion_dict(conversion_dict, word, tokenized_word)

        return conversion_dict

    def _calculate_hit_count_with_dict_and_mapping(
        self, conversion_dict: dict[str, str], char_to_token_mapping: dict[str, int]
    ) -> int:
        """指定された変換辞書とマッピングでhit countを計算する"""
        train_df, _ = get_train_valid_tsv_path(
            char_to_token_mapping=char_to_token_mapping,
            unk_token_id=0,
            use_existing_tsv=True,
            return_dataframe=True,
        )
        tokenized_df = train_df

        hit_count = 0
        total_words = 0
        for sentence, tokenized_sentence in zip(
            tokenized_df["text"], tokenized_df["tokenized_text"]
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                total_words += 1
                if (
                    self.word_to_freq_dict[word]
                    < self.word_to_freq_dict[conversion_dict[tokenized_word]]
                    and tokenized_word in conversion_dict
                ):
                    hit_count += 1

        return hit_count

    def _tournament_selection(
        self, population: List[dict[str, int]], fitness_scores: List[int]
    ) -> dict[str, int]:
        """トーナメント選択"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _crossover(
        self, parent1: dict[str, int], parent2: dict[str, int]
    ) -> dict[str, int]:
        """交叉操作"""
        child = {}
        chars = list(parent1.keys())
        crossover_point = len(chars) // 2

        for i, char in enumerate(chars):
            if i < crossover_point:
                child[char] = parent1[char]
            else:
                child[char] = parent2[char]

        return child

    def _mutate(self, individual: dict[str, int]) -> dict[str, int]:
        """突然変異操作"""
        mutated = individual.copy()
        mutations = 0
        for char in mutated:
            if random.random() < self.mutation_rate:
                old_value = mutated[char]
                mutated[char] = random.randint(0, 9)
                mutations += 1
        if mutations > 0:
            print(f"    Applied {mutations} mutations")
        return mutated

    def solve(self) -> None:
        """テストデータで予測を実行し、スコアを算出する"""
        print("\n=== Starting Test Prediction ===")
        self._solve_impl()

    def _solve_impl(self) -> None:
        """予測実装"""

        def replacer(match: re.Match) -> str:
            tokenized_word = match.group(0)
            # トークン化された単語が登録されている場合、最も確率の高い単語に変換
            if tokenized_word in self.conversion_dict:
                return self.conversion_dict[tokenized_word]
            # 登録されていない場合、最も確率の高い文字列に変換
            return "".join(
                [
                    self.token_to_most_frequent_char_mapping[token]
                    for token in tokenized_word
                ]
            )

        print("Loading test data...")
        test_path = get_test_tsv_path(
            char_to_token_mapping=self.char_to_token_mapping, unk_token_id=0
        )
        test_df = pd.read_csv(test_path, sep="\t")
        print(f"Loaded test data: {len(test_df)} samples")

        print("Applying predictions...")
        test_df["text"] = test_df["tokenized_text"].apply(
            lambda x: re.sub(r"\d+", replacer, x)
        )

        print("Saving predictions to sub.tsv...")
        test_df.to_csv("sub.tsv", sep="\t", index=False, columns=["id", "text"])

        print("Calculating accuracy...")
        accuracy = calculate_accuracy("sub.tsv")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Final accuracy percentage: {accuracy * 100:.2f}%")

    def _load_train_data(self) -> pd.DataFrame:
        """訓練データを読み込む"""
        print("start reading train data...")
        train_df, _ = get_train_valid_tsv_path(
            char_to_token_mapping={},
            unk_token_id=0,
            use_existing_tsv=True,
            return_dataframe=True,
        )
        print("done reading train data.")
        return train_df

    def _calculate_word_frequencies(self) -> dict[str, int]:
        """単語の頻度を計算する"""
        return self.train_df["text"].str.split().explode().value_counts().to_dict()

    def _create_token_to_char_mapping(self) -> dict[str, str]:
        """トークンから最も頻度の高い文字へのマッピングを作成する"""
        token_to_most_frequent_char_mapping = {}
        for char, token in self.char_to_token_mapping.items():
            if str(token) not in token_to_most_frequent_char_mapping:
                token_to_most_frequent_char_mapping[str(token)] = char
        return token_to_most_frequent_char_mapping

    def _create_conversion_dict(self) -> dict[str, str]:
        """変換辞書を作成する"""
        # トークン化されたデータを読み込む
        train_df, _ = get_train_valid_tsv_path(
            char_to_token_mapping=self.char_to_token_mapping,
            unk_token_id=0,
            use_existing_tsv=True,
            return_dataframe=True,
        )
        tokenized_df = train_df

        conversion_dict = {}
        for sentence, tokenized_sentence in tqdm(
            zip(tokenized_df["text"], tokenized_df["tokenized_text"]),
            desc="Creating conversion dict (words)",
            total=len(tokenized_df),
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                self._update_conversion_dict(conversion_dict, word, tokenized_word)

        return conversion_dict

    def _update_conversion_dict(
        self, conversion_dict: dict[str, str], word: str, tokenized_word: str
    ) -> None:
        """変換辞書を更新する"""
        if (
            tokenized_word in conversion_dict
            and conversion_dict[tokenized_word] != word
        ):
            if (
                self.word_to_freq_dict[word]
                > self.word_to_freq_dict[conversion_dict[tokenized_word]]
            ):
                conversion_dict[tokenized_word] = word
        else:
            conversion_dict[tokenized_word] = word

    def _calculate_hit_count(self) -> int:
        """ヒット数を計算する"""
        train_df, _ = get_train_valid_tsv_path(
            char_to_token_mapping=self.char_to_token_mapping,
            unk_token_id=0,
            use_existing_tsv=True,
            return_dataframe=True,
        )
        tokenized_df = train_df

        hit_count = 0
        for sentence, tokenized_sentence in tqdm(
            zip(tokenized_df["text"], tokenized_df["tokenized_text"]),
            desc="Calculating hit count (words)",
            total=len(tokenized_df),
        ):
            if not isinstance(sentence, str):
                continue
            for word, tokenized_word in zip(
                sentence.split(), tokenized_sentence.split()
            ):
                if self._is_hit(word, tokenized_word):
                    hit_count += 1

        return hit_count

    def _is_hit(self, word: str, tokenized_word: str) -> bool:
        """ヒット判定を行う"""
        return (
            self.word_to_freq_dict[word]
            < self.word_to_freq_dict[self.conversion_dict[tokenized_word]]
            and tokenized_word in self.conversion_dict
        )

    def get_results(
        self,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], dict[str, int]]:
        """結果を取得する"""
        return (
            self.conversion_dict,
            self.token_to_most_frequent_char_mapping,
            self.char_to_token_mapping,
            self.word_to_freq_dict,
        )


if __name__ == "__main__":
    solver = UnigramWithGASolver(population_size=15, generations=15, mutation_rate=0.1)
    solver.solve()
