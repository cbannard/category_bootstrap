"""
Detects noun lemmas in the postprocessed corpus that look like they were
produced by the trf tagger's lemmatizer incorrectly stripping a trailing "s"
from an English "plurale tantum" noun (a noun with no singular form, e.g.
"clothes", "scissors", "knickers") - producing a lemma that is not itself a
real word (e.g. "clothe", "scissor", "knicker").

This is the script referenced in from_tagged_corpus_to_seeds.py's comment
above the "_clothe_NOUN" -> "_clothes_NOUN" etc. corrections. Run it after
regenerating the postprocessed corpus (or against the existing one) to
re-derive/refresh that correction list - it does NOT apply any corrections
itself, it only prints candidates for manual review.

For each noun lemma that already appears in noun_selection.csv, four
mechanical conditions all have to hold for it to be printed as a candidate:

  (a) bare_count == 0
      The lemma's own surface form never occurs by itself anywhere in the
      corpus (e.g. the surface word "clothe" is never used).

  (b) plural_count / total >= 0.95
      At least 95% of the occurrences mapped to this lemma have the surface
      form lemma+"s" (e.g. "clothes"). This is NOT implied by (a) alone -
      there can be other surface variants (most commonly possessives, e.g.
      "night's", or alternate spellings, e.g. "antennae") that also never
      equal the bare lemma; (b) is what rules those out.

  (c) the bare lemma has NO noun sense in WordNet
      i.e. it is not a real English noun on its own.

  (d) lemma+"s" DOES have a noun sense in WordNet
      i.e. the "+s" form is a real dictionary word.

Only lemmas passing all four are printed. This is purely mechanical - it
does not know anything about meaning, part of speech beyond WordNet's noun
listing, or proper nouns. In particular, WordNet also lists many proper
nouns/named individuals (e.g. "Leeds", "Thomas", "Hercules", "Missis") as
having a noun sense, so some candidates will legitimately be names rather
than lemmatizer bugs - telling those apart requires eyeballing the printed
list by hand (this corpus is fully lowercased, so there's no capitalization
signal to automate that last step). See from_tagged_corpus_to_seeds.py for
which candidates were manually confirmed as real bugs and corrected, and
which were manually excluded as proper nouns.

Usage:
    python detect_fake_singular_lemmas.py
"""

import re
from collections import defaultdict, Counter

import wn
import pandas as pd

CORPUS_FILE = "manchester_input_tagged_trf_word_and_lemma_postprocessed.txt"
NOUN_SELECTION_FILE = "noun_selection.csv"
BARE_COUNT_MAX = 0          # bare lemma's own surface form must never occur
PLURAL_RATIO_MIN = 0.95     # >=95% of occurrences must be the "+s" surface form

en = wn.Wordnet("omw-en:1.4")


def has_noun_sense(word):
    return len(en.synsets(word, pos="n")) > 0


def build_lemma_surface_counts(corpus_file):
    """lemma -> Counter(surface form -> count), for tokens tagged NOUN."""
    lemma_surface = defaultdict(Counter)
    pat = re.compile(r"([^ _]+)_([^ _]+)_([^ ]+)")
    with open(corpus_file) as f:
        for line in f:
            for m in pat.finditer(line):
                surface, lemma, tag = m.group(1), m.group(2), m.group(3)
                if tag == "NOUN":
                    lemma_surface[lemma.lower()][surface.lower()] += 1
    return lemma_surface


def find_candidates(lemma_surface, noun_lemmas):
    candidates = []
    for lemma in noun_lemmas:
        if "+" in lemma or not lemma.isalpha():
            continue
        surfaces = lemma_surface.get(lemma, Counter())
        total = sum(surfaces.values())
        if total == 0:
            continue

        plural_form = lemma + "s"
        plural_count = surfaces.get(plural_form, 0)
        bare_count = surfaces.get(lemma, 0)

        if bare_count > BARE_COUNT_MAX:
            continue
        if plural_count / total < PLURAL_RATIO_MIN:
            continue
        if has_noun_sense(lemma):
            continue
        if not has_noun_sense(plural_form):
            continue

        candidates.append((lemma, plural_form, total, plural_count))

    candidates.sort(key=lambda x: -x[2])
    return candidates


def main():
    lemma_surface = build_lemma_surface_counts(CORPUS_FILE)
    nouns = pd.read_csv(NOUN_SELECTION_FILE, index_col=0)
    noun_lemmas = set(nouns["Word"].astype(str))

    candidates = find_candidates(lemma_surface, noun_lemmas)

    print(f"Found {len(candidates)} candidate(s) - review by hand before adding "
          f"any correction to from_tagged_corpus_to_seeds.py:\n")
    for lemma, plural_form, total, plural_count in candidates:
        print(f"{lemma!r:15} <- surface {plural_form!r:15} count={total}")


if __name__ == "__main__":
    main()
