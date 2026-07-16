import nltk
import wn
import pandas as pd
import re
from collections import defaultdict

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
            fi.write(line + "\n")


noun_tokens=defaultdict(int)
verb_tokens=defaultdict(int)
#tokens_tags=dict()
tokens=[]
tags=[]
filename="manchester_input_tagged_trf_word_and_lemma_postprocessed.txt"
names=["anna","anne","aran","becky","carl","caroline","dominic","gail","joel","john","julie","liz","nicole","nina","rachel","ruth","warren","wayne"]

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


d= defaultdict(int)
for i in range(nouns.shape[0]):
    lemma=str(nouns.iloc[i,0])
    #print(lemma)
    syns =  en.synsets(lemma, pos='n')
    #if len(syn) > 0:
    lem=[]
    for this_syn in syns:    
      for path in wn.taxonomy.hypernym_paths(this_syn):
         for i, ss in enumerate(path):
            lem.extend([l for l in ss.lemmas()])

    if ("physical entity" in lem):
        d[lemma] = 1
    else:
        d[lemma] = 0

nouns=nouns.merge(pd.DataFrame(d.items(),columns=["Word","Include"]),left_on='Word',right_on='Word')

nouns.to_csv("noun_selection.csv")
verbs.to_csv("verb_selection.csv")

