import json
import sys
import re
from typing import Callable

from char_category import CharCategory, character_category
from main import ByClass, NGramStats

KEYBOARD_CATEGORIES = {
    CharCategory.PLAIN,
    CharCategory.CAPITAL_LETTER,
    CharCategory.SHIFT_SPECIAL,
}


AMERICAN_ALPHANUMERIC_REGEX = re.compile(r"^[a-zA-Z0-9]$")


def is_keyboard_char(char: str) -> bool:
    return character_category(char) in KEYBOARD_CATEGORIES


def is_american_alphanumeric(char: str) -> bool:
    return AMERICAN_ALPHANUMERIC_REGEX.match(char) is not None


def rarest_char(unigrams: list, is_included: Callable[[str], bool]) -> list:
    """Return the rarest US keyboard character, according to the ngrams"""
    keyboard_unigrams = filter(lambda x: is_included(x[0][0]), unigrams)
    return min(keyboard_unigrams, key=lambda x: x[1]["count"])


def main(args: list[str]):
    if len(args) != 2:
        print(
            """Usage: python -m src/rarest_char.py <class_name> ngrams.json"
        frequency_file.json is the output of src/main.py --create-ngrams
        class_name is one of the following:
        - keyboard
        - american_alphanumeric
        """,
            sys.stderr,
        )
        sys.exit(1)
    class_name, ngram_filename = args
    with open(ngram_filename, "r", encoding="utf-8") as f:
        ngrams: dict[str, list] = json.load(f)
    if class_name == "keyboard":
        is_included = is_keyboard_char
    elif class_name == "american_alphanumeric":
        is_included = is_american_alphanumeric
    else:
        print(f"Unknown class name {class_name}", file=sys.stderr)
        sys.exit(1)
    print(rarest_char(ngrams["1"], is_included))


if __name__ == "__main__":
    main(sys.argv[1:])
