import argparse
import nltk
import wn
import pandas as pd
import re
import time
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="Build noun/verb seed candidate lists from a tagged corpus."
)
parser.add_argument(
    "--proper-noun-ratio-threshold", type=float, default=0.99,
    help=(
        "Force Include=0 for noun lemmas tagged PROPN at least this fraction "
        "of the time in the raw corpus (measured before the PROPN->NOUN "
        "retag below collapses the distinction) - i.e. proper nouns/names "
        "get excluded from the seed list, but are still retagged to NOUN so "
        "they remain valid corpus-tagged nouns/targets everywhere else in "
        "the pipeline (see category_bootstrap.py). Default 0.99 is "
        "deliberately conservative: this corpus has many physical-entity "
        "nouns that happen to share a name with a TV/story character (e.g. "
        "'teddy' is tagged PROPN ~79%% of the time, 'bear' ~53%%) - a lower "
        "threshold would wrongly exclude those from the seed list. Only "
        "lower this (e.g. --proper-noun-ratio-threshold 0.5) after "
        "reviewing which additional lemmas it would newly exclude."
    ),
)
args = parser.parse_args()

# Tallies, per lemma, how often it was tagged PROPN vs. NOUN in the RAW
# tagger output - filled in below, before the "_PROPN"->"_NOUN" substitution
# collapses that distinction. Used after the WordNet inclusion check further
# down to force Include=0 for lemmas that are proper nouns/names.
proper_noun_tag_counts = defaultdict(lambda: {"PROPN": 0, "NOUN": 0})
_propn_noun_pat = re.compile(r"([^ _]+)_([^ _]+)_(PROPN|NOUN)")

lemma_file="manchester_input_tagged_trf_word_and_lemma.txt"
output_filename="manchester_input_tagged_trf_word_and_lemma_postprocessed.txt"
with open(output_filename, 'w') as fi:
    with open(lemma_file, 'r') as infile:
        lines = infile.readlines()
        i = 0
        for line in lines:
            i = i+1
            line = line.rstrip()
            line = re.sub("gon_go_VERB na_to_([A-Z]+)","gonna_gonna_VERB",line)
            line = re.sub("got_got_VERB ta_to_([A-Z]+)","gotta_gotta_VERB",line)
            for m in _propn_noun_pat.finditer(line):
                _lemma, _tag = m.group(2).lower(), m.group(3)
                proper_noun_tag_counts[_lemma][_tag] += 1
            line = re.sub("_PROPN","_NOUN",line)
            line = re.sub("wanna_[A-Z]+","wanna_VERB",line)
            line = re.sub("hasta_[A-Z]+","hasta_VERB",line)
            line = re.sub("needta_[A-Z]+","needta_VERB",line)
            line = re.sub("oughta_[A-Z]+","oughta_VERB",line)
            line = re.sub("sposta_[A-Z]+","sposta_VERB",line)
            line = re.sub("hafta_[A-Z]+","hafta_VERB",line)
            line = re.sub("useta_[A-Z]+","useta_VERB",line)
            line = re.sub("hadta_[A-Z]+","hadta_VERB",line)
            line = re.sub("([^ \\_]+)_AUX","\\1_VERB",line)
            # be/do/have are excluded from the VERB category entirely - a
            # blanket exclusion covering both auxiliary and lexical/main-verb
            # uses (e.g. "have" meaning possess, "do" as a main verb). Retag
            # any occurrence (whichever tag it ended up with above) back to
            # AUX, so all_tagged_nouns_verbs=True mode - which reads the tag
            # directly and doesn't consult seed lists - never counts these
            # three lemmas as verbs either. This mirrors the existing
            # INCLUDE_human=0 exclusion for be/do/have in
            # verb_inclusion.xlsx, which only covers the seed-based modes.
            line = re.sub(r"([^ \_]+)_(be|do|have)_(VERB|AUX)", r"\1_\2_AUX", line)
            line = re.sub("'ll_([^ \\_]+)'ll_[A-Z]+","_\\1_NOUN 'll_will_VERB",line)
            line = re.sub("([a-z]+)@l_([^ ]+)","\\1@l_\\1@l_NOUN",line)
            line = re.sub("([^ \\_]+)_([^ \\_]+)_NOUN 's\\_'s_PART","\\1's_\\1_NOUN",line)
            line = re.sub("([^ \\_]+)_([^ \\_]+)_PRON 's\\_'s_PART","\\1's_\\1_PRON",line)
            line = re.sub("ca_ca_VERB","ca_can_VERB",line)
            line = re.sub("wo_wo_VERB","wo_will_VERB",line)
            line = re.sub("sha_sha_VERB","sha_shall_VERB",line)
            line = re.sub(",_,_PUNCT ","",line)
            # We extracted a list of conjoined elements that are tagged as NOUN.
            # The following items were judged not to be NOUNs and so are retagged.
            line = re.sub("night_night_NOUN","night_night_X",line)
            line = re.sub("a_lot_of_NOUN","a_lot_of_X",line)
            line = re.sub("lots_of_NOUN","lots_of_X",line)
            line = re.sub("happy_birthday_NOUN","happy_birthday_X",line)
           
            line = re.sub("see_saw_marjorie_daw_NOUN","see_saw_marjorie_daw_X",line)
            line = re.sub("thank_you_NOUN","thank_you_X",line)
            line = re.sub("wakie_wakie_NOUN","wakie_wakie_X",line)
            line = re.sub("(o\\'clock)_NOUN","\\1_X",line)
            line = re.sub("(none)_NOUN","\\1_X",line)
            line = re.sub("(pretend)_NOUN","\\1_X",line)
            line = re.sub("(-)_NOUN","\\1_X",line)
            line = re.sub("(upsidedown)_NOUN","\\1_X",line)

            # The trf tagger's lemmatizer treats a trailing "s" as a regular
            # plural marker and strips it even for English "plurale tantum"
            # nouns that have no singular form - it never occurs bare in the
            # corpus, isn't a real word on its own, and WordNet only
            # recognizes the "+s" form (e.g. "clothes"_"clothe"_NOUN, where
            # "clothe" the noun doesn't exist, but "clothes" does and is
            # correctly classified as a physical entity/apparel). Detected by
            # scanning for lemmas where (a) the bare lemma's own surface form
            # never occurs, (b) essentially all occurrences are the "+s"
            # surface form, (c) the bare lemma has no WordNet noun sense, and
            # (d) "+s" does. Retag the lemma (not the surface form, which is
            # already correct) back to the "+s" form so it merges with any
            # correctly-lemmatized occurrences and gets a correct WordNet
            # inclusion decision below.
            #
            # NOT included here: "leed"/"thoma"/"hercule"/"missi" (->
            # "leeds"/"thomas"/"hercules"/"missis") - these hit the same
            # detection rule, but they're proper nouns/character names
            # (Leeds the city, Thomas the Tank Engine, Hercules, "the
            # missis"), and WordNet also has entries for those as named
            # individuals. "Fixing" the lemma there would flip them to
            # Include=1 for the wrong reason - pulling a specific named
            # individual into what's meant to be a common-noun seed list.
            # There's no signal in this (fully lowercased) corpus to tell
            # those apart automatically, so they're left as-is (excluded)
            # rather than auto-corrected.
            line = re.sub("_clothe_NOUN","_clothes_NOUN",line)
            line = re.sub("_knicker_NOUN","_knickers_NOUN",line)
            line = re.sub("_thank_NOUN","_thanks_NOUN",line)
            line = re.sub("_scissor_NOUN","_scissors_NOUN",line)
            line = re.sub("_tight_NOUN","_tights_NOUN",line)
            line = re.sub("_underpant_NOUN","_underpants_NOUN",line)
            line = re.sub("_after_NOUN","_afters_NOUN",line)
            line = re.sub("_gymnastic_NOUN","_gymnastics_NOUN",line)
            line = re.sub("_dramatic_NOUN","_dramatics_NOUN",line)
            line = re.sub("_lazybone_NOUN","_lazybones_NOUN",line)
            line = re.sub("_aerobic_NOUN","_aerobics_NOUN",line)
            line = re.sub("_oasi_NOUN","_oasis_NOUN",line)
            line = re.sub("_acrobatic_NOUN","_acrobatics_NOUN",line)
            line = re.sub("_logistic_NOUN","_logistics_NOUN",line)
            line = re.sub("_tiddlywink_NOUN","_tiddlywinks_NOUN",line)

            fi.write(line + "\n")


noun_tokens=defaultdict(int)
verb_tokens=defaultdict(int)
#tokens_tags=dict()
tokens=[]
tags=[]
filename="manchester_input_tagged_trf_word_and_lemma_postprocessed.txt"
names=["anna","anne","aran","becky","carl","caroline","dominic","gail","joel","john","julie","liz","nicole","nina","rachel","ruth","warren","wayne"]
# add mummy, daddy etc?

with open(filename) as file:
        for line in file:
            tokens.append("{")
            tags.append("BOS")
            line_array = line.split()
            for element in line_array:
                la=re.match("[^ ]+\\_([^ ]+)\\_([^ ]+)",element)
                w=la.group(1)
                if w in names:
                    w = "pname"
                t=la.group(2)
                tokens.append(w)
                tags.append(t)
                if re.match("NOUN",t):
                    noun_tokens[str.lower(w)] += 1
                if re.match("VERB",t):
                    verb_tokens[str.lower(w)] += 1
            tokens.append("}")
            tags.append("EOS")

sorted_noun_counts=sorted(noun_tokens.items(), key=lambda item: item[1], reverse=True)
sorted_verb_counts=sorted(verb_tokens.items(), key=lambda item: item[1], reverse=True)
sorted_noun_tokens=list(zip(*sorted_noun_counts))[0]
sorted_verb_tokens=list(zip(*sorted_verb_counts))[0]
tokens.insert(0,"{")
tokens.insert(len(tokens),"}")
token_count=len(tokens)

sorted_noun_counts=sorted(noun_tokens.items(), key=lambda item: item[1], reverse=True)
sorted_noun_tokens=list(zip(*sorted_noun_counts))[0]
nouns=pd.DataFrame(data=sorted_noun_counts,columns=["Word","Count"])
verbs=pd.DataFrame(data=sorted_verb_counts,columns=["Word","Count"])

nltk.download('wordnet')
en=wn.Wordnet('omw-en:1.4')

from nltk.corpus import wordnet as nltk_wn

from collections import defaultdict

# Verb inclusion is now based on human judgments in verb_inclusion.xlsx rather
# than WordNet. A verb is included (1) if its lemma appears in the "lemma"
# column with INCLUDE_human == 1; any lemma not present in the sheet, or
# present with INCLUDE_human != 1, is excluded (0).
verb_inclusion_df = pd.read_excel("verb_inclusion.xlsx")
human_include_lookup = dict(zip(verb_inclusion_df["lemma"].astype(str), verb_inclusion_df["INCLUDE_human"]))

# Sanity check: every lemma in verb_inclusion.xlsx should actually occur among
# the verbs extracted from the corpus. A lemma that doesn't match anything is
# most likely a typo or a stale entry from a previous corpus/tagset.
corpus_verb_set = set(verbs["Word"].astype(str))
missing_lemmas = sorted(set(human_include_lookup) - corpus_verb_set)
if missing_lemmas:
    raise ValueError(
        f"{len(missing_lemmas)} lemma(s) in verb_inclusion.xlsx do not occur in the "
        f"corpus verb list: {missing_lemmas}"
    )

d = defaultdict(int)
for i in range(verbs.shape[0]):
    lemma = str(verbs.iloc[i, 0])
    d[lemma] = 1 if human_include_lookup.get(lemma) == 1 else 0

verbs=verbs.merge(pd.DataFrame(d.items(),columns=["Word","Include"]),left_on='Word',right_on='Word')


# This loop does one WordNet lookup + hypernym-path traversal per distinct
# noun lemma in the corpus, which for a full-size corpus can be thousands of
# lemmas and take a long time - print periodic progress so a long-running
# job doesn't look hung with no output.
num_nouns_to_check = nouns.shape[0]
print(f"Checking WordNet ('physical entity' hypernym) inclusion for {num_nouns_to_check} candidate noun(s)...")
noun_check_start = time.time()
progress_every = max(1, num_nouns_to_check // 50) if num_nouns_to_check else 1

d= defaultdict(int)
for i in range(num_nouns_to_check):
    lemma=str(nouns.iloc[i,0])
    # The corpus (and therefore this lemma) joins multiword compounds with
    # "+" (e.g. "ice+cream"), but WordNet's own multiword entries are joined
    # with "_" (e.g. "ice_cream") - "+" never matches anything in WordNet, so
    # every multiword lemma was silently falling through to Include=0
    # regardless of whether WordNet actually knows the compound. Query
    # WordNet using the "_"-joined form, but keep `lemma` (with "+") as the
    # dict key / eventual "Word" value, since that's what has to keep
    # matching the corpus's own "+"-joined lemmas downstream (see
    # category_bootstrap.py's noun_set/verb_set seed matching).
    wordnet_lookup_lemma = lemma.replace("+", "_")
    syns =  en.synsets(wordnet_lookup_lemma, pos='n')
    lem=[]
    for this_syn in syns:
      for path in wn.taxonomy.hypernym_paths(this_syn):
         for _, ss in enumerate(path):
            lem.extend([l for l in ss.lemmas()])

    if ("physical entity" in lem):
        d[lemma] = 1
    else:
        d[lemma] = 0

    if (i + 1) % progress_every == 0 or (i + 1) == num_nouns_to_check:
        elapsed = time.time() - noun_check_start
        print(f"  ...{i + 1}/{num_nouns_to_check} nouns checked ({elapsed:.0f}s elapsed)")

print(f"WordNet noun inclusion check done in {time.time() - noun_check_start:.0f}s.")

nouns=nouns.merge(pd.DataFrame(d.items(),columns=["Word","Include"]),left_on='Word',right_on='Word')

# Force Include=0 for lemmas that are proper nouns/names, per
# proper_noun_tag_counts collected above - this overrides whatever WordNet
# said, since WordNet also has entries for many named individuals (e.g.
# "thomas", "leeds", "hercules", "missis" all resolve to a WordNet noun
# sense and would otherwise get Include=1 for the wrong reason - a named
# individual, not a common-noun category). See --proper-noun-ratio-threshold
# above for why this is a ratio rather than "ever tagged PROPN" - many
# ordinary physical-entity nouns in this corpus (teddy, bear, fox, dolly...)
# are ALSO sometimes tagged PROPN (character-name confusion) without being
# proper nouns themselves.
def _propn_ratio(lemma):
    counts = proper_noun_tag_counts.get(lemma)
    if not counts:
        return 0.0
    total = counts["PROPN"] + counts["NOUN"]
    return (counts["PROPN"] / total) if total else 0.0

nouns["ProperNounRatio"] = nouns["Word"].map(_propn_ratio)
is_proper_name = nouns["ProperNounRatio"] >= args.proper_noun_ratio_threshold
nouns.loc[is_proper_name, "Include"] = 0
print(
    f"Forced Include=0 for {int(is_proper_name.sum())} noun lemma(s) tagged "
    f"PROPN >= {args.proper_noun_ratio_threshold:.0%} of the time "
    f"(--proper-noun-ratio-threshold={args.proper_noun_ratio_threshold})."
)

nouns.to_csv("noun_selection.csv")
verbs.to_csv("verb_selection.csv")

