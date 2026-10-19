"""Text normalisation and fact matching shared by the benchmark checker and the evaluator.

A "fact" is a list of acceptable variants (e.g. ["19.5%", "19.5 percent"]); it
matches a text if any variant occurs with no letter or digit directly before
or after it, so "18" does not match inside "2018".
"""

import re

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-"})
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    text = text.translate(_DASHES).translate(_QUOTES).lower()
    return re.sub(r"\s+", " ", text).strip()


def variant_in(variant: str, text: str) -> bool:
    pattern = r"(?<![0-9a-z])" + re.escape(normalize(variant)) + r"(?![0-9a-z])"
    return re.search(pattern, normalize(text)) is not None


def fact_in(fact: list[str] | str, text: str) -> bool:
    variants = [fact] if isinstance(fact, str) else fact
    return any(variant_in(v, text) for v in variants)


def fact_recall(facts: list, text: str) -> float:
    if not facts:
        return 1.0
    return sum(fact_in(f, text) for f in facts) / len(facts)
