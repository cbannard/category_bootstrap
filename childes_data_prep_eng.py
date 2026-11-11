#!/usr/bin/env python3
"""Prepare the English (Manchester) CHILDES corpus and POS-tag it with spaCy.

Converted from ChildesDataPrep_Eng.ipynb. Steps:

1. (optional, --download) fetch ManchesterCorpus.zip from Google Drive with
   gdown and unzip it.
2. Walk the corpus directory, join CHAT lines that were wrapped across several
   physical lines (writing a ``.rem`` copy next to each ``.cha``/``.cex`` file),
   and extract adult/caregiver utterances, cleaned of CHAT markup.
3. Tag and lemmatize every utterance with spaCy's ``en_core_web_trf`` model and
   write one utterance per line as space-separated ``word_lemma_POS`` tokens -
   the ``manchester_input_tagged_trf_word_and_lemma.txt`` file that
   from_tagged_corpus_to_seeds.py reads. ``--no-lemma`` gives the old
   ``word_POS`` format instead.

Usage:
    python3 childes_data_prep_eng.py --download
    python3 childes_data_prep_eng.py --corpus-dir Manchester/ --output manchester_input_tagged_trf_word_and_lemma.txt
    python3 childes_data_prep_eng.py --report-odd-chars --no-tag   # just inspect the cleaned text
"""

import argparse
import copy
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict

# Google Drive file ID of ManchesterCorpus.zip.
MANCHESTER_ZIP_GDRIVE_ID = "1YmiXKfAks2KnuSfRNpgfVnvnWBcsFsQy"
MANCHESTER_ZIP_NAME = "ManchesterCorpus.zip"

# Speaker tiers kept from the transcripts (adults/caregivers, not the target child).
SPEAKER_RE = r"\*(ADU|BRO|DAD|FAT|FRI|GRA|INV|JUI|MOT|NIC|RAC|WAY)\:"


def recurse_dirs(dir):

    outtext = []
    outlist = []
    corpora = os.listdir(dir)
    for corpus in corpora:
        path = dir + corpus
        dirbin = []
        print(corpus)
        for d_path, d_names, f_names in os.walk(path):

            if any(fname.endswith('.cha') for fname in os.listdir(d_path)):

                # for f in_names:
                # if f.endswith('cha'):

                subdir = d_path + "/"
                remove_embedded_returns(subdir)
                for f in os.listdir(subdir):
                    if f.endswith('.rem'):

                        outtext.append(get_file_text(subdir + f, f))
    return outtext


def get_file_text(file, filename):

    thistext = []
    previousline = "START"
    with open(file, 'r') as infile:
        lines = infile.readlines()
        i = 0
        for line in lines:
            i = i + 1
            if not re.match(r".*xxx.*", line) and not re.match(r".*www.*", line):

                if re.match(SPEAKER_RE, line.rstrip('\n')):

                    speaker = re.search(SPEAKER_RE, line)[0]
                    line = deal_with_variant_form_exceptions(line)
                    line = deal_with_variant_forms(line)
                    line = deal_with_marked_errors_etc(line)
                    # some residual rules
                    line = re.sub(r"\+\/+\.", ".", line)
                    line = re.sub(r"\+\"\/? ", "", line)
                    line = re.sub(r"\+\.\.\. ", "", line)
                    line = re.sub(r"\+\/+\? ", "", line)
                    # line = re.sub("\+","",line)
                    line = re.sub(r"\+\/+", "", line)
                    line = re.sub(r"\+\"\/", "", line)
                    line = re.sub(r"\+\"", "", line)
                    line = deal_with_special_characters(line)
                    line = re.sub(r"\&-um", "", line)
                    line = re.sub(r"\&-er", "", line)
                    line = re.sub(r"\&mm", "", line)
                    line = re.sub(r"\&ah", "", line)
                    line = re.sub(r"\&a:h", "", line)
                    line = re.sub(r"\&-eh", "", line)
                    line = re.sub(r"\(\.\) ", "", line)
                    line = re.sub(r"\.\.\.", ".", line)
                    line = re.sub(r"\++ ", "", line)
                    line = re.sub(r" \++", "", line)
                    line = re.sub(r"\@[^ l]+", "", line)
                    line = re.sub(r"0[^ ]*", "", line)
                    line = re.sub(r";", "", line)
                    # next rule needs consideration
                    # line = re.sub("([^ ]+)\@[a-z]","\\1",line)

                    if i <= len(lines):
                        # and lines[i].strip().startswith('%mor:'):
                        response = line.split(':', 1)[1]
                        response = re.sub(r"\:", "", response)
                        response_as_list = response.split()

                        if len(response_as_list) > 0:
                            # and "xxx" not in response_as_list:
                            outp = " ".join(response_as_list)
                            if not re.search(r"\&", outp) and re.search(r"[a-zA-Z]", outp):
                                thistext.append(outp)
    return thistext


def remove_embedded_returns(my_dir):
    files = os.listdir(my_dir)
    for file in files:
        if (('.cha' in file
             or '.cex' in file)
            and '.rem' not in file
                and '._' not in file):
            with open(my_dir + file, 'r') as infile:
                inlist = infile.readlines()
                for i in range(len(inlist)):
                    inlist[i] = inlist[i].split()
                    if inlist[i] == []:
                        inlist[i] = ['binthis']
                while ['binthis'] in inlist:
                    inlist.remove(['binthis'])
                for j in range(5):
                    for i in range(len(inlist)):
                        if (len(inlist[i]) > 0
                            and inlist[i][0][0] in ['%', '*', '@', '%']
                            and i < len(inlist) - 2
                            and inlist[i+1][0][0] not in ['%', '*', '@', '%']
                                and inlist[i+1] != ['binthis']):
                            inlist[i] = copy.deepcopy(
                                inlist[i]) + copy.deepcopy(inlist[i+1])
                            inlist[i+1] = ['binthis']
                        if inlist[i] == []:
                            inlist[i] = ['binthis']
                    while ['binthis'] in inlist:
                        inlist.remove(['binthis'])

            with open(my_dir + file + '.rem', 'w') as outfile:
                for line in inlist:
                    outfile.write(' '.join(line) + '\n')


def deal_with_variant_form_exceptions(textline):

    textline = textline.replace('ahem ach [: I don’t]', 'I don’t')
    textline = textline.replace('Uh no [: don’t know]', 'don’t know')
    textline = textline.replace('goinga [: going to]', 'gonna')
    textline = textline.replace('gon [: going to]', 'gonna')
    textline = textline.replace('deres [: there is]', 'there’s')
    textline = textline.replace('don [: do not]', 'don’t')
    textline = textline.replace('hed+up [: fed up]', 'fed up')
    textline = textline.replace('hull+up [: full up]', 'full up')
    textline = textline.replace('won\'t [: willn\'t]', 'won\'t')
    textline = textline.replace('willn\'t', 'won\'t')
    textline = textline.replace(
        'creabids [: creamy porridge]', 'creabids')
    textline = textline.replace('fire engine ', 'fire+engine ')

    return textline


def deal_with_variant_forms(textline):
    textline = textline.split(" ")
    newtextline = []
    for item in textline:
        item = item.replace('(', '')
        item = item.replace(')', '')

        if item == '[:':
            break
        else:
            newtextline.append(item.lower())
    return " ".join(newtextline)


def deal_with_marked_errors_etc(textline):
    # Note: this is a plain string replace, not a regex, so it looks for the
    # literal characters \]\[ (with backslashes), as in the original notebook.
    textline = textline.replace('\\]\\[', '] [')
    textline = re.sub(r'([^ ]),', '\\1 ,', textline)
    textline = re.sub(r',([^ ])', ', \\1', textline)
    textline = re.sub(r'\<', '', textline)
    textline = re.sub(r'\>', '', textline)
    textline = re.sub(r'\[[^\[\]]*\] ?', '', textline)
    textline = re.sub(r'\(\.\) ?', '', textline)

    textline = textline.replace('0is ', '')
    return textline


def deal_with_special_characters(textline):
    # outline = re.sub("\, ","",textline)
    outline = re.sub("„ ", ", ", textline)
    return outline


def download_corpus(dest_dir="."):
    """Fetch ManchesterCorpus.zip from Google Drive (needs `pip install gdown`) and unzip it."""
    zip_path = os.path.join(dest_dir, MANCHESTER_ZIP_NAME)
    subprocess.run(["gdown", MANCHESTER_ZIP_GDRIVE_ID, "-O", zip_path], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def report_odd_characters(all_input):
    """Print every utterance containing a character outside the expected set (a sanity check on the cleanup)."""
    for s in all_input:
        for i in s:
            if re.search(r"[^ a-zA-Z\?,\.\'\_\-\!\@\^\"\“\”]", i):
                print(i)


def tag_utterances(all_input, output_file, model="en_core_web_trf", with_lemma=True):
    """POS-tag each utterance with spaCy and write one line per utterance.

    Tokens are written as `word_lemma_POS` (or `word_POS` if with_lemma is False).
    """
    import spacy  # imported here so the other steps don't need spaCy installed

    nlp = spacy.load(model)
    with open(output_file, 'w') as fi:
        for j in range(len(all_input)):
            for utt in all_input[j]:
                doc = nlp(str(utt))
                if with_lemma:
                    tokens = [w.text + "_" + w.lemma_ + "_" + w.pos_ for w in doc]
                else:
                    tokens = [w.text + "_" + w.pos_ for w in doc]
                fi.write(" ".join(tokens) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", default="Manchester/",
                        help="directory containing one subdirectory per child corpus (default: Manchester/)")
    parser.add_argument("--output", default=None,
                        help="tagged output file, overwritten if it exists (default: "
                             "manchester_input_tagged_trf_word_and_lemma.txt, or "
                             "manchester_input_tagged_trf.txt with --no-lemma)")
    parser.add_argument("--no-lemma", action="store_true",
                        help="write word_POS tokens (the notebook's original format) instead of word_lemma_POS")
    parser.add_argument("--download", action="store_true",
                        help=f"download and unzip {MANCHESTER_ZIP_NAME} from Google Drive first (requires gdown)")
    parser.add_argument("--report-odd-chars", action="store_true",
                        help="print cleaned utterances that still contain unexpected characters")
    parser.add_argument("--no-tag", action="store_true",
                        help="skip spaCy tagging (e.g. with --report-odd-chars, to just inspect the cleanup)")
    parser.add_argument("--spacy-model", default="en_core_web_trf",
                        help="spaCy model to tag with (default: en_core_web_trf; "
                             "install with `python -m spacy download en_core_web_trf`)")
    args = parser.parse_args()

    if args.output is None:
        args.output = ("manchester_input_tagged_trf.txt" if args.no_lemma
                       else "manchester_input_tagged_trf_word_and_lemma.txt")

    if args.download:
        download_corpus(os.path.dirname(os.path.normpath(args.corpus_dir)) or ".")

    corpus_dir = args.corpus_dir if args.corpus_dir.endswith("/") else args.corpus_dir + "/"
    if not os.path.isdir(corpus_dir):
        sys.exit(f"Corpus directory not found: {corpus_dir} (use --download to fetch it)")

    all_input = recurse_dirs(corpus_dir)
    print(f"Extracted {sum(len(s) for s in all_input)} utterances from {len(all_input)} transcripts")

    if args.report_odd_chars:
        report_odd_characters(all_input)

    if not args.no_tag:
        tag_utterances(all_input, args.output, args.spacy_model, with_lemma=not args.no_lemma)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
