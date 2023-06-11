import argparse
import json
import random
import sys
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import (
    TextIO,
    Iterator,
    Iterable,
    Optional,
    cast,
    Union,
    TypeVar,
    Callable,
)
from langdetect import detect_langs, LangDetectException
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from unicodedata import normalize

from char_category import (
    CharCategory,
    character_category,
    TypedCharAndCategory,
    CONTROL_KEY_STRINGS,
    NAVIGATION_KEY_STRINGS,
    ACTION_KEY_STRINGS,
)


@dataclass
class WarcHeader:
    lines: list[str]

    def is_conversion(self) -> bool:
        for line in self.lines:
            if line.startswith("WARC-Type: conversion"):
                return True
        return False


@dataclass
class WarcSection:
    header: WarcHeader
    content: str


def warc_sections(lines_raw: TextIO, file: Path) -> Iterator[WarcSection]:
    lines = tqdm(lines_raw, desc=f"Reading {file.name}", unit=" lines").__iter__()
    line = next(lines, "")
    while True:
        header_lines = []
        if line == "":
            return
        while not line.startswith("WARC/1.0"):
            line = next(lines, "")
            if line == "":
                return
        while line != "\n":
            header_lines.append(line)
            line = next(lines, "")
            if line == "":
                return
        header = WarcHeader(header_lines)
        content = ""
        line = next(lines, "")
        while not (line == "" or line.startswith("WARC/1.0")):
            content = content + line
            line = next(lines, "")
        yield WarcSection(header, content)


def split_wet(split_wet_dir: Path) -> int:
    if not split_wet_dir.exists() or not split_wet_dir.is_dir():
        print(f"Error: {split_wet_dir} is not a directory", file=sys.stderr)
        return 1
    wet_files = split_wet_dir.glob("*.wet")
    for wet_file in wet_files:
        with wet_file.open("r", encoding="utf-8") as f:
            for section in warc_sections(f, wet_file):
                if not section.header.is_conversion():
                    continue
                if section.content == "":
                    continue
                try:
                    langs = detect_langs(section.content)
                except LangDetectException:
                    langs = []
                if len(langs) == 0:
                    lang_name = "no-languages-detected"
                else:
                    lang = langs[0]
                    lang_name = lang.lang if lang.prob > 0.9 else "uncertain"
                lang_file = split_wet_dir / ("language_" + lang_name + ".txt")
                with lang_file.open("a", encoding="utf-8") as lang_f:
                    lang_f.write(section.content)
                    lang_f.write("\n")
    return 0


@contextmanager
def open_text(output_filename: Optional[str], mode: str) -> Iterable[TextIO]:
    """Open the output file, or the appropriate standard stream
    if output_filename is None.

    Throws if there is an error opening the output file"""
    if mode not in ["w", "a", "r"]:
        raise ValueError(f"Invalid text file mode {mode}")
    f = (
        open(output_filename, mode, encoding="utf-8")
        if output_filename is not None
        else (sys.stdin if mode == "r" else sys.stdout)
    )
    try:
        yield f
    finally:
        if output_filename is not None:
            f.close()


def type_char(char: str, prev_category: CharCategory) -> TypedCharAndCategory:
    """Return a sequence of strings representing how to type the character on an
    idealized keyboard with sticky modifier keys given that the previous character
    was of the given category. Return the symbols to type and the category of the
    character"""
    cat = character_category(char)
    prefix: list[str] = []
    if cat != prev_category:
        prefix = [cat.value]

    if cat == CharCategory.UNICODE_4:
        return TypedCharAndCategory(prefix + list(f"{ord(char) :04X}"), cat)
    if cat == CharCategory.UNICODE_8:
        return TypedCharAndCategory(prefix + list(f"{ord(char) :08X}"), cat)
    return TypedCharAndCategory(prefix + [char], cat)


def type_section(section: list[str]) -> list[str]:
    """Replace each pseudo-character in the section with a sequence of strings
    representing how to type it on the keyboard"""
    typed = []
    prev_category = CharCategory.PLAIN
    for char in section:
        if char in {"BEGIN", "END"}:
            typed.append(char)
            continue
        typed_char, prev_category = type_char(char, prev_category)
        typed.extend(typed_char)
    return typed


def list_a_section(s: str) -> list[str]:
    return ["BEGIN"] + list(s) + ["END"]


PROB_EXTRA_KEY = 0.01
PROB_CONTROL_KEY = 0.3
PROB_ACTION_KEY = 0.1


def add_extra_keys_to_section(section: list[str]) -> list[str]:
    """Add navigation and control key sequences to the section."""
    with_extra_keys = section.copy()
    num_extra_keys = int(len(section) * PROB_EXTRA_KEY)
    for _ in range(num_extra_keys):
        insertion_position = random.randint(1, len(with_extra_keys) - 1)
        if (
            insertion_position < len(with_extra_keys) - 2
            and random.random() < PROB_CONTROL_KEY
        ):
            control_key = random.choice(CONTROL_KEY_STRINGS)
            with_extra_keys.insert(insertion_position, control_key + "_DOWN")
            up_position = random.randint(
                insertion_position + 1, len(with_extra_keys) - 1
            )
            with_extra_keys.insert(up_position, control_key + "_UP")
            continue
        if random.random() < PROB_ACTION_KEY:
            action_key = random.choice(ACTION_KEY_STRINGS)
            if action_key == "BACKSPACE_KEY":
                normal_key = random.choice(section)
                with_extra_keys.insert(insertion_position, normal_key)
            with_extra_keys.insert(insertion_position, action_key)
            continue
        navigation_key = random.choice(NAVIGATION_KEY_STRINGS)
        with_extra_keys.insert(insertion_position, navigation_key)
    return with_extra_keys


def create_ngrams(
    max_ngram_class: int,
    type_characters: bool,
    add_extra_keys: bool,
    rare_character_threshold: int,
    input_filename: Optional[str],
    output_filename: Optional[str],
) -> int:
    try:
        bar: tqdm
        with open_text(input_filename, "r") as input_file:
            chars = input_file.read()
            sections = [normalize("NFKC", s) for s in chars.split("\n\n\n\n")]
            # noinspection PyUnusedLocal
            chars = None  # free memory
            split_sections = process_map(
                list_a_section,
                sections,
                chunksize=max(1, len(sections) // 1000),
                desc="Splitting sections",
                unit=" sections",
            )
            # noinspection PyUnusedLocal
            sections = None  # free memory
            if add_extra_keys:
                split_sections = process_map(
                    add_extra_keys_to_section,
                    split_sections,
                    chunksize=max(1, len(split_sections) // 1000),
                    desc="Adding extra keys",
                    unit=" sections",
                )
            if type_characters:
                split_sections = process_map(
                    type_section,
                    split_sections,
                    chunksize=max(1, len(split_sections) // 1000),
                    desc="Typing sections",
                    unit=" sections",
                )
            raw_counts = return_counts(max_ngram_class, split_sections)
            bar = tqdm(total=5)
            bar.set_description("Reorganizing counts")
            reorganized_counts = reorganize_counts(raw_counts)
            bar.update(1)
            bar.set_description("Removing rare characters")
            without_rare_characters = combine_rare_characters(
                reorganized_counts, rare_character_threshold
            )
            bar.update(1)
            bar.set_description("Adding probabilities")
            with_probabilities = add_probabilities(without_rare_characters)
            bar.update(1)
            bar.set_description("Converting to lists")
            counts = convert_to_lists(with_probabilities)
        bar.update(1)
        bar.set_description("Writing to file")
        with open_text(output_filename, "w") as output_file:
            output_file.write(
                json.dumps(
                    counts,
                    ensure_ascii=False,
                    indent=2,
                    separators=(", ", ": "),
                    default=vars,
                )
            )
        bar.update(1)
        return 0
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def add_to_counts(counts: dict, to_add: deque[str]):
    current_level = counts
    for char in to_add:
        if char not in current_level:
            current_level[char] = {}
        current_level[char]["COUNT"] = current_level[char].get("COUNT", 0) + 1
        current_level = current_level[char]


class ChunkCounter:
    def __init__(self, max_ngram_class: int):
        self.max_ngram_class = max_ngram_class

    def __call__(self, sections: list[list[str]]) -> dict:
        counts = {}
        for section in sections:
            if not section:
                continue
            window = deque(section[: self.max_ngram_class], maxlen=self.max_ngram_class)
            add_to_counts(counts, window)
            for char in section[self.max_ngram_class :]:
                window.append(char)
                add_to_counts(counts, window)
        return counts


def chunker(it: Iterable, chunk_size: int) -> Iterable:
    iterator = iter(it)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def return_counts(
    max_ngram_class: int, sections: list[list[str]], use_tqdm=True
) -> dict:
    """Return the counts of n-grams in the sections

    counts[first_char]["COUNT"] = number of times first_char,
    appears in the corpus
    counts[first_char][second_char]["COUNT"] = number of times first_char,
    second_char appears in the corpus
    counts[first_char][second_char][third_char]["COUNT" = number of times
    the substring first_char, second_char, third_char appears in
    the corpus

    Parameters
    ----------
    max_ngram_class : int
        The maximum n-gram class to count. Unigrams would be 1, bigrams 2, etc.
    sections : list[str]
        The sections to count n-grams in. Each section should be a string. They
        matter for counting the first and last characters of sections.
    use_tqdm : bool
        Whether to use tqdm to show a progress bar. Defaults to True. Here
        to make test output cleaner.
    """
    chunks = [c for c in chunker(sections, max(1, len(sections) // 500))]
    separate_counts = process_map(
        ChunkCounter(max_ngram_class),
        chunks,
        desc="Counting n-grams",
        chunksize=max(1, len(chunks) // 2000),
        unit=" section chunks",
        disable=not use_tqdm,
    )
    counts = {}
    for section_count in tqdm(
        separate_counts, desc="Combining counts", unit=" chunks", disable=not use_tqdm
    ):
        add_items(counts, section_count)

    return counts


def add_items(to_modify: dict, to_add: dict):
    """Add the counts from to_add to to_modify"""
    for key, value in to_add.items():
        if key not in to_modify:
            to_modify[key] = 0 if key == "COUNT" else {}
        if key == "COUNT":
            to_modify[key] += value
            continue
        add_items(to_modify[key], value)


@dataclass(frozen=True, eq=True, order=True)
class NGramStats:
    """Stats for an individual n-gram"""

    # The number of times the n-gram appears in the corpus
    count: int
    # Total of all counts for n-grams of the same class
    total: int
    # The modal probability of the n-gram appearing in the corpus using a
    # Beta(0.5, 0.5) prior
    probability: float
    # 1%-ile probability of this n-gram using a Beta(0.5, 0.5) prior
    min_probability: float
    # 99%-ile probability of this n-gram using a Beta(0.5, 0.5) prior
    max_probability: float


T = TypeVar("T")

# A dict of dicts of T keyed by n-gram class and then n-gram
ByClass = dict[int, dict[tuple[str, ...], T]]


def reorganize_counts(counts: dict) -> ByClass[int]:
    """Reorganize the counts from return_counts to be a dict of dicts of
    counts keyed by n-gram class and then n-gram"""
    reorganized_counts: ByClass[int] = {}
    reorganize_counts_for_prefix(counts, reorganized_counts, cast(tuple[str], ()))
    return reorganized_counts


def reorganize_counts_for_prefix(
    sub_counts: dict[str, Union[dict, int]],
    reorganized_counts: ByClass[int],
    prefix: tuple[str, ...],
) -> None:
    for key, value in sub_counts.items():
        if isinstance(value, int):
            if len(prefix) not in reorganized_counts:
                reorganized_counts[len(prefix)] = {}
            reorganized_counts[len(prefix)][prefix] = value
        elif isinstance(key, str):
            reorganize_counts_for_prefix(value, reorganized_counts, prefix + (key,))
        else:
            # Shouldn't ever happen
            raise ValueError(
                f"Unexpected key {key} when reorganizing dict for prefix {prefix}"
            )


def combine_rare_characters(
    reorganized_counts: ByClass[int], rare_character_threshold: int
) -> ByClass[int]:
    """Return the counts with rare characters combined into a single RARE_CHARACTER
    symbol."""
    combined_counts: ByClass[int] = defaultdict(lambda: defaultdict(int))
    if 1 not in reorganized_counts:
        raise ValueError("No unigrams in counts when combining rare characters")
    rare_characters = {
        k[0]
        for k, count in reorganized_counts[1].items()
        # Skip rare special symbols like BEGIN and END
        if count < rare_character_threshold and len(k[0]) == 1
    }
    for ngram_class, ngrams in reorganized_counts.items():
        for ngram, count in ngrams.items():
            new_ngram = tuple(
                "RARE_CHARACTER" if char in rare_characters else char for char in ngram
            )
            combined_counts[ngram_class][new_ngram] += count
    return combined_counts


def add_probabilities(
    reorganized_counts: ByClass[int], use_tqdm=True
) -> ByClass[NGramStats]:
    """Add probabilities to the counts"""
    from scipy.stats import beta

    num_ngrams = sum(len(ngrams) for ngrams in reorganized_counts.values())
    progress_bar = tqdm(
        total=num_ngrams,
        disable=not use_tqdm,
        unit=" n-gram classes",
        desc="Adding probabilities",
    )
    with_probs: ByClass[NGramStats] = {}
    for ngram_class, ngrams in reorganized_counts.items():
        with_probs[ngram_class] = {}
        total = sum(ngrams.values())
        for ngram, count in ngrams.items():
            freq_prob = beta(count + 0.5, total - count + 0.5)
            with_probs[ngram_class][ngram] = NGramStats(
                count=count,
                total=total,
                probability=count / total if total > 0 else 0,
                min_probability=freq_prob.ppf(0.01),
                max_probability=freq_prob.ppf(0.99),
            )
            progress_bar.update(1)
    return with_probs


def convert_to_lists(
    by_class: ByClass[T],
) -> dict[int, list[tuple[tuple[str], T]]]:
    return {
        ngram_class: list(
            sorted(ngrams.items(), key=lambda i: (i[1], i[0]), reverse=True)
        )
        for ngram_class, ngrams in by_class.items()
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-wet-dir",
        type=str,
        help="Split the WET files "
        "(from common crawl) in this directory into language-specific "
        "input files. Ignore other options",
    )
    parser.add_argument(
        "--create-ngrams",
        type=int,
        help="Create n-grams of this size.  "
        "Assumes files are broken into sections by four consecutive "
        "newlines (as happens with --split-wet-dir). Reads all text "
        "into memory from --input-file and writes JSON to --output-file. "
        "We can rewrite it if we need to handle large corpora.",
    )
    parser.add_argument(
        "--type-characters",
        action="store_true",
        help="Replace characters in the input file with sequences"
        "representing how to type them on a keyboard.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file name. " "Will be overwritten. Standard output if omitted.",
    )
    parser.add_argument(
        "--input-file", type=str, help="Input file name." "Standard input if omitted."
    )
    parser.add_argument(
        "--rare-character-threshold",
        type=int,
        default=0,
        help="If a character appears fewer than this many times, replace it with a "
        "special RARE_CHARACTER symbol. This does not apply to special symbols like "
        "BEGIN and END. Default 0, which means characters cannot be rare characters.",
    )
    parser.add_argument(
        "--add-extra-keys",
        action="store_true",
        help="Add random control keys like shift, alt, control, and windows logo to "
        "the input file as well as navigation keys like arrow keys and home/end.",
    )
    args = parser.parse_args(argv[1:])
    if args.split_wet_dir is not None:
        return split_wet(Path(args.split_wet_dir))
    if args.create_ngrams is not None:
        return create_ngrams(
            args.create_ngrams,
            args.type_characters,
            args.add_extra_keys,
            args.rare_character_threshold,
            args.input_file,
            args.output_file,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

# main.py
#
# Copyright 2023-06 Eric Moyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
