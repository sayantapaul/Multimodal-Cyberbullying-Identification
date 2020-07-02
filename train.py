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

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', required=True, help='Name of model to train [bilstm, rcnn, bilstm_rcnn]')
parser.add_argument('--dataset', '-d', help='Path to dataset, Default: dataset/vine_labeled_cyberbullying_data.csv', default='dataset/vine_labeled_cyberbullying_data.csv')
parser.add_argument('--text_embedding', default='dataset/comment_embedding.csv', help='Path to Text embedding, Default: dataset/comment_embedding.csv')
parser.add_argument('--video_features', default='dataset/video_features.csv', help='Path to Text embedding, Default: dataset/video_features.csv')
parser.add_argument('--hidden_size', '-hs', default=512, help='Hidden Size of BiLSTM | Default: 512', type=int)
parser.add_argument('--no_comments', default=25, help='Number of Comments to include | Default: 25', type=int)

parser.add_argument('--batch_size', '-b', default=64, help='Batch Size | Default: 64', type=int)
parser.add_argument('--epochs', '-e', default=100, help='No of Epochs | Default: 100', type=int)
parser.add_argument('--logs', '-l', default='logs', help='Path to Logs (weights, tensorboard) | Default: logs_[model_name]', type=str)
parser.add_argument('--no_classes', '-c', default=2, help='Number of Classes | Default: 2', type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, help='Learning Rate | Default: 0.001', type=float)
parser.add_argument('--train_val_split', '-tvs', default=0.2, help='Train vs Validation Split | Default: 0.2', type=float)

parser.add_argument('--check_build', action='store_true', help='Check if the model can be built or not')

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
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
model.summary()

if args.check_build:
    exit()

reader = ReadData(dataset=args.dataset, text_embedding_path=args.text_embedding,
                  video_feature_path=args.video_features, data_shape=inputs, train_val_split=args.train_val_split)

train_generator = reader.generator(batch_size=args.batch_size)
val_generator = reader.generator_val(batch_size=args.batch_size)

log_dir = args.logs + '_' + args.model
logging = TrainValTensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'),
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)

print('Training on {} samples and Validating on {} samples.'.format(reader.train_size, reader.val_size))

model.fit_generator(generator=train_generator, steps_per_epoch=int(reader.train_size/args.batch_size),
                    validation_data=val_generator, epochs=args.epochs, validation_steps=int(reader.val_size/args.batch_size),
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
