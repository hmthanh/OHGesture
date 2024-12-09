import yaml
from pprint import pprint
from easydict import EasyDict
import argparse
import io
import tqdm
import numpy as np
import librosa
from gensim.models import KeyedVectors
import csv
import os
from praatio import textgrid
import fasttext


def parse_args():
    parser = argparse.ArgumentParser(description='OHGesture')
    parser.add_argument('--wav', type=str, required=True, default='./003_Neutral_2_x_1_0.wav')
    parser.add_argument('--text', type=str, required=True, default='./003_Neutral_2_x_1_0.csv')
    parser.add_argument('--word2vec_model', required=True, type=str, default="../fasttext/crawl-300d-2M.vec")
    parser.add_argument('--subword_model', type=str, default="../fasttext/crawl-300d-2M-subword")

    args = parser.parse_args()
    return args


def text_grid2tsv(file):
    # Read the TextGrid file using praatio
    tg = textgrid.openTextgrid(file, False)

    # Get the first tier (assuming that's what we want based on original code)
    tier = tg.getTier(tg.tierNames[0])

    # Create output filename by replacing .TextGrid with .tsv
    output_file = file.replace('.TextGrid', '.tsv')

    # Write entries to TSV
    with open(output_file, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        # Iterate through entries in the tier
        for entry in tier.entries:
            start_time, end_time, label = entry

            # Skip empty labels (similar to original)
            if label == '':
                continue

            tsv_writer.writerow([start_time, end_time, label])


def load_tsv_aligned(file):
    sentence = []
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])

    return sentence


def load_csv_aligned(file):
    sentence = []
    with open(os.path.join(file), "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if len(line) == 5:
                start, end, raw_word, type, speaker = line

                if type == "words":
                    start = float(start)
                    end = float(end)
                    sentence.append([start, end, raw_word])

    return sentence


def word2vec(sentence, word2vec_model, n_frames, fps, csv_file):
    tensor_vec = np.zeros([n_frames, 300])

    for start, end, word in sentence:
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        try:
            # Get the vector for the word
            vector = word2vec_model.get_vector(word.lower())

        except KeyError:
            with open("./log.txt", "a") as f:
                f.write(f"Inside file {csv_file}, Token {word} not found in word2vec model\n")
            print(f"Token {word} not found in word2vec model")
            vector = np.zeros(300)

            # Handle missing words
            # print(f"Keyword '{word}' not found in {tsv_file}")
            # try:
            #     # Find the most similar word in the vocabulary
            #     similar_word = word2vec_model.most_similar(normalized_word, topn=1)[0][0]
            #     print(f"Using similar word '{similar_word}' for '{word}'")
            #     vector = word2vec_model.get_vector(similar_word)
            # except KeyError:
            #     # If no similar word is found, use a zero vector
            #     print(f"No similar word found for '{word}'. Using a zero vector.")
            #     vector = np.zeros(300)

        tensor_vec[start_frame:end_frame, :] = vector

    print("embedding shape", np.shape(tensor_vec))
    return tensor_vec


if __name__ == '__main__':
    '''
    python word_embedding.py --wav=./003_Neutral_2_x_1_0.wav --text=003_Neutral_2_x_1_0.csv --word2vec_model=../fasttext/crawl-300d-2M.vec
    '''
    args = parse_args()
    pprint(args)
    fps = 20

    word2vec_model = KeyedVectors.load_word2vec_format(args.word2vec_model, binary=False)
    # subword_model = fasttext.load_model(args.subword_model)

    wav, sr = librosa.load(args.wav, sr=16000)
    audio_length_seconds = len(wav) / sr
    print(wav.shape)

    # tsv_file = os.path.join(args.src, f"{wav_file[:-4]}.tsv")
    # align_sentence = load_tsv_aligned(tsv_file)

    align_sentence = load_csv_aligned(args.text)

    n_frames = int(audio_length_seconds * fps)
    print(f"Audio length: {audio_length_seconds} -> {audio_length_seconds * fps}")
    sentence_vec = word2vec(align_sentence, word2vec_model, n_frames, fps, args.text)
    print(np.shape(sentence_vec), " -> saving ", f"{args.wav[:-4]}.npy")
    np.save(f"{args.wav[:-4]}.npy", sentence_vec)
