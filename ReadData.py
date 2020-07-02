import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import re
import cv2

#from keras.applications.xception import Xception
#from keras.applications.vgg16 import VGG16

def convert_str_to_array(array_str):
    array_str = array_str.replace('[', '')
    array_str = array_str.replace(']', '')
    array_str = array_str.replace(' ', '')
    return np.fromstring(array_str, sep=', ')

def convert_str_to_array_vid(array_str):
    array_str = array_str.replace('[ ', '')
    array_str = array_str.replace(' ]', '')
    #print(array_str)
    #print(len(array_str))

    return np.fromstring(array_str, sep=' ')


class VideoToFrames:
    def __init__(self, video_path, no_frames=30):
        self.reader = cv2.VideoCapture(video_path)

        self.no_frames = no_frames
        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames != 0:
            self.interval = int(self.total_frames/self.no_frames)

    def convert(self):
        frames = []
        frame_number = 1
        while len(frames) < self.no_frames:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.reader.read()

            try:
                if frame == None:
                    break
            except ValueError:
                pass

            frames.append(cv2.resize(frame, (48, 48)))
            frame_numb
        #print(self.embeddings.head())er += self.interval

        return frames

class ReadData:
    def __init__(self, dataset, text_embedding_path, video_feature_path, url_to_postid='dataset/urls_to_postids.txt', video_dir='dataset/videoFolder', data_shape=[(99, 512), (30, 512)], train_val_split=0.2):
        print('Reading Dataset ..')
        self.dataset = pd.read_csv(dataset)
        self.dataset = self.dataset.sample(frac=1.0).reset_index(drop=True)
        print('Done.')

        print('Reading Text Embeddings ..')
        self.embeddings = pd.read_csv(text_embedding_path)
        self.embeddings.embedding = self.embeddings.embedding.apply(convert_str_to_array)
        print('Done.')

        print('Reading Video Features ..')
        self.video_features = pd.read_csv(video_feature_path, sep='\t')
        #self.video_features = self.video_features[['links', 'features']]
        #self.video_features = self.video_features.drop_duplicates()

        self.valid_videolinks = list(self.video_features.links)
        self.video_features.features = self.video_features.features.apply(convert_str_to_array)

        self.video_features = self.video_features.groupby('links')

        print('Done.')

        self.training = self.dataset.head(int(len(self.dataset)*(1-train_val_split))).reset_index(drop=True)
        self.validation = self.dataset.tail(int(len(self.dataset)*(train_val_split))).reset_index(drop=True)

        self.train_size = len(self.training)
        self.val_size = len(self.validation)
        self.text_shape = data_shape[0]
        self.video_shape = data_shape[1]

        #self.url_to_postid = pd.read_csv(url_to_postid)
        #self.video_dir = video_dir

        #self.image_feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
        #self.image_feature_extractor.summary()

    def sent2vec(self, sent, embed_size=512):
        embed = self.embeddings[self.embeddings.comment == sent]
        if embed.empty:
            return np.zeros(embed_size)

        return list(embed.embedding)[0]

    def post_embedding(self, comments, max_len=99, dim=512):
        vector = []
        for i, comment in enumerate(comments):
            if i < max_len:
                #try:
                vec = self.sent2vec(comment, dim)
                vector.append(vec)
                #except Exception as e:
                #    pass
        vectors = []
        vectors += list(vector)
        padding_len = max_len - len(vectors)
        for _ in range(padding_len):
            vectors.append(np.zeros(dim))

        return np.array(vectors)

    def get_comments(self, row, i):
        comments = []
        for j in range(1, self.text_shape[0]+1):
            comm = row['column{}'.format(j)][i]
            try:
                comm = [x.strip() for x in re.findall('\:\:.*?(.*)\(created', comm, re.MULTILINE | re.DOTALL)]
                if comm == []:
                    break
                else:
                    comments += comm
            except TypeError:
                pass

        return comments

    def get_video_features(self, link):
        group = list(self.video_features.get_group(link).features)
        features = []
        for i in range(self.video_shape[0]):
            features.append(group[i])

        #print(np.array(features).shape)

        return np.array(features)

    def generator(self, batch_size=8):
        while True:
            no_batches = int(self.train_size/batch_size)
            for i in range(no_batches):
                start_index = i*batch_size
                text_features, video_features, batch_y = [], [], []
                for j in range(i, i+batch_size):
                    row = self.training.iloc[[j]]
                    if row.videolink[j] in self.valid_videolinks:
                        comments = self.get_comments(row, j)
                        text = self.post_embedding(comments, self.text_shape[0], self.text_shape[1])

                        video = self.get_video_features(row.videolink[j])

                        y = []
                        if row['question2'][j] == 'noneBll':
                            y.append([1, 0])
                        else:
                            y.append([0, 1])

                        text_features.append(text)
                        video_features.append(video)
                        batch_y.append(y)

                x, y = [np.array(text_features), np.array(video_features)], np.array(batch_y)
                y = np.reshape(y, (y.shape[0], 2))
                yield x, y

    def generator_val(self, batch_size=8):
        while True:
            no_batches = int(self.val_size/batch_size)
            for i in range(no_batches):
                start_index = i*batch_size
                text_features, video_features, batch_y = [], [], []
                for j in range(i, i+batch_size):
                    row = self.validation.iloc[[j]]
                    if row.videolink[j] in self.valid_videolinks:
                        comments = self.get_comments(row, j)
                        text = self.post_embedding(comments, self.text_shape[0], self.text_shape[1])

                        video = self.get_video_features(row.videolink[j])

                        y = []
                        if row['question2'][j] == 'noneBll':
                            y.append([1, 0])
                        else:
                            y.append([0, 1])

                        text_features.append(text)
                        video_features.append(video)
                        batch_y.append(y)

                x, y = [np.array(text_features), np.array(video_features)], np.array(batch_y)
                y = np.reshape(y, (y.shape[0], 2))
                yield x, y

    def next_batch(self, i, batch_size=64):
        batch_x, batch_y = [], []
        for j in range(i, i+batch_size):
            row = self.training.iloc[[j]]
            comments = self.get_comments(row, j)
            x = self.post_embedding(comments, self.text_shape[0], self.text_shape[1])
            y = []
            if row['question2'][j] == 'noneBll':
                y.append([1, 0])
            else:
                y.append([0, 1])

            batch_x.append(x)
            batch_y.append(y)

        x, y = np.array(batch_x), np.array(batch_y)

        return x, y


if __name__ == '__main__':
    reader = ReadData(dataset='dataset/vine_labeled_cyberbullying_data.csv', text_embedding_path='dataset/comment_embedding.csv',
                      video_feature_path='dataset/video_features.csv', data_shape=[(25, 512), (30, 1024)], train_val_split=0.)

    for x, y in reader.generator():
        if len(x[0].shape) < 3:
            for item in x[0]:
                print(np.array(item).shape)
        if len(x[1].shape) < 3:
            for item in x[1]:
                print(np.array(item).shape)
        print(x[0].shape, x[1].shape, y.shape)
