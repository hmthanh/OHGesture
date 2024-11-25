import os
import sys
import nltk
from contractions import fix
import argparse
from pprint import pprint
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('punkt_tab')


def parse_args():
    parser = argparse.ArgumentParser(description='OHGesture')
    parser.add_argument('--src', type=str, default='./txt')
    parser.add_argument('--dest', type=str, default='./txt_processed')
    parser.add_argument('--word2vec_model', type=str, default="./fasttext/crawl-300d-2M.vec")

    args = parser.parse_args()
    return args


def expand_contractions_nltk(text, word2vec_model, file):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Expand contractions
    expanded_tokens = [fix(token) for token in tokens]

    def fix_not(token):
        # Check if the token is "n't" and return its expanded form
        return "not" if token == "n't" else token

    expanded_tokens_fixed = [fix_not(token) for token in expanded_tokens]

    # Rechecks
    for token in expanded_tokens_fixed:
        try:
            word2vec_model.get_vector(token)
        except KeyError:
            with open("./data/train/log.txt", "a") as f:
                f.write(f"Inside file {file}, Token {token} not found in word2vec model\n")
            print(f"Token {token} not found in word2vec model")

    # Rejoin tokens into a sentence
    expanded_text = ' '.join(expanded_tokens_fixed)
    return expanded_text


def process_txt_contractions(file, word2vec_model):
    sentence_list = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            raw_text = line.strip()
            sentences = nltk.sent_tokenize(raw_text)

            for sentence in sentences:
                sentence_list.append(sentence)

    sentence_result = [expand_contractions_nltk(sen, word2vec_model, file) for sen in sentence_list]
    return " ".join(sentence_result)
    # for sentence in sentence_list:
    #     print("Original:", sentence)
    #     expanded_text = expand_contractions_nltk(sentence)
    #     print("Expanded:", expanded_text, "\n\n")


if __name__ == '__main__':
    """
    python process_txt_contractions.py --src="./data_full/train/txt" --dest="./data_full/train/corpus" --word2vec_model=./fasttext/crawl-300d-2M.vec
    """
    # Example usage
    # text = "I'll go to the market, and we'll see what happens."
    args = parse_args()
    pprint(args)

    transcribes = os.listdir(args.src)
    files = [os.path.join(args.src, f) for f in transcribes]
    word2vec_model = KeyedVectors.load_word2vec_format(args.word2vec_model, binary=False)

    for file in files:
        print("Processing", file)
        txt_processed = process_txt_contractions(file, word2vec_model)

        name = file.split("/")[-1][:-4]
        lab_file = os.path.join(args.dest, f"{name}.lab")
        with open(lab_file, "w") as f:
            f.write(txt_processed)
            print(f"Saved {lab_file}")
