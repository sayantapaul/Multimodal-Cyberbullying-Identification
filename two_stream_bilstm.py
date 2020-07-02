from keras.layers import Dense, Input, Bidirectional, LSTM, dot, concatenate, Activation
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model

import tensorflow as tf

class TwoStreamBiLSTM:
    def __init__(self, hidden_size=512, no_classes=2):
        self.hidden_size = hidden_size
        self.no_classes = no_classes

    def build(self, input_shape=((25, 512), (30, 1536))):
        input_text =  Input(shape=input_shape[0])
        input_video =  Input(shape=input_shape[1])

        text = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='text_bilstm-1')(input_text)
        text = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='text_bilstm-2')(text)

        video = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='video_bilstm-1')(input_video)
        video = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='video_bilstm-2')(video)

        attention = dot([text, video], axes=[2, 2])
        attention = Activation('softmax', name='softmax')(attention)

        context = dot([attention, video], axes=[2, 1], name='video_context')
        text_combined_context = concatenate([context, text])

        x = Bidirectional(LSTM(self.hidden_size*2, return_sequences=True, kernel_initializer='glorot_uniform'))(text_combined_context)

        x1 = GlobalAveragePooling1D()(x)
        #x2 = GlobalMaxPooling1D()(x)

        #x = concatenate([x1, x2])
        x = Dense(self.hidden_size*2, kernel_initializer='glorot_uniform')(x1)
        x = LeakyReLU(0.2)(x)

        x = Dropout(rate=0.2)(x)

        output = Dense(self.no_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)

        return Model(inputs=[input_text, input_video], outputs=output)


if __name__ == '__main__':
    model = TextVideoClassifier()
    model = model.build()

    model.summary()
