import re
from enum import Enum
from typing import NamedTuple

PLAIN_CHARACTERS = re.compile(r"[-a-z0-9 =;'\[\t\n/,.`]|]")
CAPITAL_LETTERS = re.compile(r"[A-Z]")
SHIFT_SPECIALS = re.compile(r"[!@#$%^&*()_+{}|:\"<>?~£€©®]")


class CharCategory(Enum):
    """The category of a character for typing purposes - the value is the name symbol
    to type to make the keyboard enter that category"""

    PLAIN = "PLAIN"
    CAPITAL_LETTER = "CAPITAL_LETTER"
    SHIFT_SPECIAL = "SHIFT_SPECIAL"
    CONTROL_NAV_ACTION = "CONTROL_NAV_ACTION"
    UNICODE_4 = "UNICODE_4"
    UNICODE_8 = "UNICODE_8"


def character_category(char: str) -> CharCategory:
    """Return the category of a character"""
    if len(char) > 1:
        if char in NAVIGATION_ACTION_AND_CONTROL_KEY_STRING_SET:
            return CharCategory.CONTROL_NAV_ACTION
        raise ValueError(f"Unknown character {char}")
    if PLAIN_CHARACTERS.match(char):
        return CharCategory.PLAIN
    if CAPITAL_LETTERS.match(char):
        return CharCategory.CAPITAL_LETTER
    if SHIFT_SPECIALS.match(char):
        return CharCategory.SHIFT_SPECIAL
    code_point = ord(char)
    if code_point < 0x10000:
        return CharCategory.UNICODE_4
    return CharCategory.UNICODE_8


class TypedCharAndCategory(NamedTuple):
    chars: list[str]
    category: CharCategory


CONTROL_KEY_STRINGS = [
    "CTRL_KEY",
    "ALT_KEY",
    "SHIFT_KEY",
    "SUPER_KEY",
]
CONTROL_KEY_STRING_SET = {k + "_UP" for k in CONTROL_KEY_STRINGS} | {
    k + "_DOWN" for k in CONTROL_KEY_STRINGS
}
NAVIGATION_KEY_STRINGS = [
    "ESCAPE_KEY",
    "PAGE_UP_KEY",
    "PAGE_DOWN_KEY",
    "END_KEY",
    "HOME_KEY",
    "LEFT_KEY",
    "UP_KEY",
    "RIGHT_KEY",
    "DOWN_KEY",
]
NAVIGATION_KEY_STRING_SET = set(NAVIGATION_KEY_STRINGS)
ACTION_KEY_STRINGS = [
    "INSERT_KEY",
    "DELETE_KEY",
    "BACKSPACE_KEY",
]
ACTION_KEY_STRING_SET = set(ACTION_KEY_STRINGS)
NAVIGATION_ACTION_AND_CONTROL_KEY_STRING_SET = (
    NAVIGATION_KEY_STRING_SET | ACTION_KEY_STRING_SET | CONTROL_KEY_STRING_SET
)

# char_category.py
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
