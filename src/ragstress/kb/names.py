"""Pseudo-word name generation with pronounceability filtering (SPEC §4.1)."""

import random
import re

# Consonant onsets, nuclei, and codas for natural-sounding pseudo-words
ONSETS = [
    "b",
    "c",
    "d",
    "f",
    "g",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "z",
    "br",
    "dr",
    "gr",
    "kr",
    "pr",
    "tr",
    "pl",
    "kl",
    "fl",
    "st",
    "sp",
    "qu",
    "th",
]
NUCLEI = ["a", "e", "i", "o", "u", "ae", "ea", "ia", "io", "ou"]
CODAS = ["", "k", "l", "m", "n", "r", "s", "t", "x", "v", "th", "nd", "nt", "rk", "st", "sh"]

CONSONANT_WHITELIST_TRIPLES = {
    "str",
    "spr",
    "scr",
    "spl",
    "thr",
    "shr",
    "nch",
    "rch",
    "dst",
    "rst",
    "lth",
    "rth",
    "mph",
    "nts",
    "nds",
    "kst",
}

VOWELS = set("aeiouy")

# Real company/brand blocklist to avoid accidental collisions
BLOCKED_REAL_NAMES = {
    "google",
    "apple",
    "microsoft",
    "amazon",
    "meta",
    "facebook",
    "oracle",
    "cisco",
    "intel",
    "nvidia",
    "tesla",
    "ibm",
    "sony",
    "samsung",
    "dell",
    "hp",
    "uber",
    "airbnb",
    "netflix",
    "stripe",
    "adobe",
    "salesforce",
    "openai",
    "anthropic",
    "twitter",
    "spacex",
    "boeing",
    "lockheed",
    "toyota",
    "ford",
    "pfizer",
    "moderna",
    "walmart",
    "target",
}

POLICY_NOUNS = [
    "Travel Policy",
    "Remote Work Policy",
    "Equipment Policy",
    "Expense Policy",
    "Leave Policy",
    "Security Policy",
    "Conduct Policy",
    "Compliance Policy",
    "Relocation Policy",
    "Training Policy",
    "Health and Safety Policy",
    "Procurement Policy",
]

PRODUCT_PREFIXES = [
    "Router",
    "Switch",
    "Sensor",
    "Gateway",
    "Optimizer",
    "Accelerator",
    "Analyzer",
    "Processor",
    "Controller",
    "Transceiver",
    "Actuator",
    "Modem",
]

FACILITY_NOUNS = [
    "Data Center",
    "Innovation Lab",
    "Logistics Hub",
    "Operations Center",
    "Research Center",
    "Distribution Center",
    "Fabrication Plant",
    "Regional Facility",
    "Testing Center",
    "Assembly Plant",
]


def is_pronounceable(word: str) -> bool:
    """Check pronounceability:

    - Has at least one vowel
    - No 3+ consecutive consonants unless in whitelist
    """
    lower = word.lower()
    if not any(c in VOWELS for c in lower):
        return False

    # Check for consecutive consonants of length 3 or more
    consonant_runs = re.findall(r"[^aeiouy\s\d_-]+", lower)
    for run in consonant_runs:
        if len(run) >= 3:
            # Check if any valid 3-gram matches whitelist
            valid = False
            for i in range(len(run) - 2):
                triple = run[i : i + 3]
                if triple in CONSONANT_WHITELIST_TRIPLES:
                    valid = True
                    break
            if not valid:
                return False
    return True


def generate_pseudo_word(rng: random.Random) -> str:
    """Generate a single pseudo-word that passes the pronounceability filter."""
    for _ in range(100):
        syllables = rng.choice([2, 3])
        parts = []
        for _ in range(syllables):
            onset = rng.choice(ONSETS)
            nucleus = rng.choice(NUCLEI)
            coda = rng.choice(CODAS)
            parts.append(f"{onset}{nucleus}{coda}")
        word = "".join(parts).capitalize()
        if 4 <= len(word) <= 12 and is_pronounceable(word):
            if word.lower() not in BLOCKED_REAL_NAMES:
                return word
    # Fallback guaranteed pronounceable pseudo-word
    return f"Vandor{rng.randint(10, 99)}"


def generate_entity_name(kind: str, rng: random.Random, existing_names: set[str]) -> str:
    """Generate a unique entity name adhering to FR-KB-1."""
    for _ in range(500):
        base_word = generate_pseudo_word(rng)
        if kind == "Policy":
            noun = rng.choice(POLICY_NOUNS)
            name = f"{base_word} {noun}"
        elif kind == "Product":
            model_num = rng.choice(range(100, 990, 10))
            # e.g., Dravix-340 or Dravix Router-340
            name = f"{base_word}-{model_num}"
        elif kind == "Facility":
            noun = rng.choice(FACILITY_NOUNS)
            name = f"{base_word} {noun}"
        else:
            name = f"{base_word} Entity"

        if name not in existing_names:
            existing_names.add(name)
            return name

    # Guaranteed fallback
    fallback = f"Vandor-{len(existing_names) + 1} {kind}"
    existing_names.add(fallback)
    return fallback
