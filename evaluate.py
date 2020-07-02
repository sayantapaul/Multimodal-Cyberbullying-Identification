from TextVideoClassifier.two_stream_bilstm import TwoStreamBiLSTM
from TextVideoClassifier.recurrent_cnn import RecurrentCNN
from TextVideoClassifier.bilstm_recurrent_cnn import BiLSTMRecurrentCNN

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

from ReadData import ReadData

import os
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', required=True, help='Name of model to train [bilstm, rcnn]')
parser.add_argument('--weights', '-w', help='Weights of trained model')
parser.add_argument('--dataset', '-d', help='Path to dataset, Default: dataset/vine_labeled_cyberbullying_data.csv', default='dataset/vine_labeled_cyberbullying_data.csv')
parser.add_argument('--text_embedding', default='dataset/comment_embedding.csv', help='Path to Text embedding, Default: dataset/comment_embedding.csv')
parser.add_argument('--video_features', default='dataset/video_features.csv', help='Path to Text embedding, Default: dataset/video_features.csv')
parser.add_argument('--hidden_size', '-hs', default=512, help='Hidden Size of BiLSTM | Default: 512', type=int)

parser.add_argument('--batch_size', '-b', default=64, help='Batch Size | Default: 64', type=int)
parser.add_argument('--no_classes', '-c', default=1, help='Number of Classes | Default: 1', type=int)
parser.add_argument('--no_comments', default=25, help='Number of Comments to include | Default: 25', type=int)

args = parser.parse_args()

hidden_size = args.hidden_size

if args.model == 'bilstm':
    inputs = [(args.no_comments, 512), (30, 1536)]
    model = TwoStreamBiLSTM(hidden_size, args.no_classes)
elif args.model == 'rcnn':
    inputs = [(args.no_comments, 512), (30, 1536)]
    model = RecurrentCNN(no_filters=hidden_size, no_classes=args.no_classes)
elif args.model == 'bilstm_rcnn':
    inputs = [(args.no_comments, 512), (30, 1536)]
    model = BiLSTMRecurrentCNN(hidden_size, no_classes=args.no_classes)

model = model.build(inputs)
model.load_weights(args.weights)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

reader = ReadData(dataset=args.dataset, text_embedding_path=args.text_embedding,
                  video_feature_path=args.video_features, data_shape=inputs, train_val_split=1.)

results = []
labels = []

prog_bar = tqdm(total=int(reader.val_size/args.batch_size))

num_batches = int(reader.val_size/args.batch_size)

i = 0

for x, y in reader.generator_val(batch_size=args.batch_size):
    label = list(y)
    result = list(model.predict(x))

    for res in label:
        labels.append(res[0])

    for res in result:
        if res > 0.5:
            results.append(1)
        else:
            results.append(0)

    if i > num_batches:
        break

    i += 1

    prog_bar.update(1)

print(classification_report(labels, results, labels=[0, 1]))
