from PIL import Image

# for json
import json

# for glob file search
import glob

# for fuzzy string compariosn
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# for encoding strings
from zlib import crc32

import logging

from random import randint


logging.getLogger().setLevel(logging.ERROR)
COORDINATE_PATH = 'ba_dataset/SROIE2019/0325updated.task1train(626p)'
TAG_PATH = 'ba_dataset/SROIE2019/0325updated.task2train(626p)'
IMG_PATH = COORDINATE_PATH

TAG_TO_IDX = {'': 0, 'company': 1, 'date': 2, 'address': 3, 'total': 4}
IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}


def get_filenames(path, suffix):
    path = path + '/' if not path.endswith('/') else path
    files = glob.glob(path + '*.' + suffix)
    for idx, file in enumerate(files):
        files[idx] = file.split("/")[-1].replace('.txt', '').replace('.jpg', '')
    return files


def get_text_filenames(path):
    return get_filenames(path, 'txt')


def get_image_filenames(path):
    return get_filenames(path, 'jpg')


def prepare_data(filename):
    tags = read_tags(filename)

    im = read_image_file(filename)
    width, height = im.size

    coordinates = read_normalized_coordinates(filename, width, height)
    coordinate_inputs = [coordinates[i][:9] for i in range(len(coordinates))]
    coordinate_texts = [coordinates[i][9] for i in range(len(coordinates))]

    coordinate_tags = match_coordinate_tags(coordinate_texts, tags)
    return tags, coordinate_inputs, coordinate_texts, coordinate_tags


def get_all_filenames():
    image_files = get_image_filenames(IMG_PATH)
    # print('found {} image files'.format(len(image_files)))

    coordinate_files = get_text_filenames(COORDINATE_PATH)
    # print('found {} files with coordinate data'.format(len(coordinate_files)))

    tag_files = get_text_filenames(TAG_PATH)
    # print('found {} files with tag data'.format(len(tag_files)))

    filenames = list(set(image_files) & set(coordinate_files) & set(tag_files))
    filenames.sort()
    return filenames


def prepare_training_data():
    filenames = get_all_filenames()
    print('found {} files with coordinate and tag data'.format(len(filenames)))

    training_size = int(len(filenames) * 0.8)
    print('using {} files for training'.format(training_size))

    training_data = []
    for i in range(training_size):
        tags, coordinate_inputs, coordinate_texts, coordinate_tags = prepare_data(filenames[i])
        training_data.append((coordinate_inputs, coordinate_tags))

    return training_data


def get_random_test_file():
    filenames = get_all_filenames()
    print('found {} files with coordinate and tag data'.format(len(filenames)))

    test_size = int(len(filenames) * 0.2)
    print('using {} files for testing'.format(test_size))
    
    return filenames[randint(test_size + 1, len(filenames) - 1)]


def read_text_file(path, filename):
    path = path + '/' if not path.endswith('/') else path
    with open(path + filename + '.txt') as file:
        text = file.read()
    return text


def read_text_file_lines(path, filename):
    path = path + '/' if not path.endswith('/') else path
    lines = []
    with open(path + filename + '.txt') as file:
        for line in file:
            lines.append(line.rstrip('\n'))
    return lines


def read_coordinates(filename, path='ba_dataset/SROIE2019/0325updated.task1train(626p)'):
    text = read_text_file_lines(path, filename)
    coordinates = []
    for line in text:
        tokens = line.split(',')
        line_coordinates = list(map(int, tokens[0:8]))
        line_text = ','.join(tokens[8:])
        line_coordinates.append(line_text)
        coordinates.append(line_coordinates)
    return coordinates


def read_normalized_coordinates(filename, width, height, path='ba_dataset/SROIE2019/0325updated.task1train(626p)'):
    coordinates = read_coordinates(filename, path=path)
    for line in coordinates:
        for x in range(0, 8, 2):
            line[x] /= width
        for x in range(1, 8, 2):
            line[x] /= height
        line.append(line[8])
        line[8] = normalize_text(line[8])
    return coordinates


def read_tags(filename, path='ba_dataset/SROIE2019/0325updated.task2train(626p)'):
    return json.loads(read_text_file(path, filename))


def read_image_file(filename, path='ba_dataset/SROIE2019/0325updated.task1train(626p)'):
    path = path + '/' if not path.endswith('/') else path
    # return cv2.imread(path + filename + '.jpg', 0)
    return Image.open(path + filename + '.jpg')


def match_coordinate_tags(coordinate_texts, tags):

    tags_reverted = {v: k for k, v in tags.items()}
    tag_values = list(tags.values())
    coordinate_tags = []
    for text in coordinate_texts:
        # text = text.replace('*', '_').replace('%', '_').replace(':', '_').replace('=', '_').
        # replace('-', '_').replace('(', '_').replace('@', '_').replace('^', '_').replace('/', '_')
        tag_guess = process.extractOne(text, tag_values, scorer=fuzz.partial_ratio, score_cutoff=90)
        tag = ''
        if tag_guess is not None:
            tag = tags_reverted[tag_guess[0]]
        coordinate_tags.append(encode_tag(tag))
    return coordinate_tags


def encode_tag(tag):
    return TAG_TO_IDX[tag]


def decode_tag(tag_idx):
    return IDX_TO_TAG[tag_idx]


def normalize_text(input_text, encoding="utf-8"):
    # see https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
    return float(crc32(input_text.encode(encoding)) & 0xffffffff) / 2**32


def combine_predicted_tags(texts, tags):
    company = ''
    date = ''
    address = ''
    total = ''

    for i in range(len(tags)):
        if tags[i] == 1 and not company.endswith(texts[i]):
            company += ' ' + texts[i]
        if tags[i] == 2 and not date.endswith(texts[i]):
            date += ' ' + texts[i]
        if tags[i] == 3 and not address.endswith(texts[i]):
            address += ' ' + texts[i]
        if tags[i] == 4 and not total.endswith(texts[i]):
            total += ' ' + texts[i]

    return {'company': company.strip(), 'date': date.strip(), 'address': address.strip(), 'total': total.strip()}
