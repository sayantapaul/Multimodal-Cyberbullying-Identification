from keras.layers import Dense, Input, Bidirectional, LSTM, dot, concatenate, Activation
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.layers import Dropout, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import add

from keras import backend as K
from keras.models import Model

class BiLSTMRecurrentCNN:
    def __init__(self, hidden_size=512, no_classes=2):
        self.hidden_size = hidden_size
        self.no_classes = no_classes

    def recurrent_block(self, input_layer):
        out_filter = self.hidden_size*2

        x1 = Conv1D(filters=out_filter, kernel_size=1, border_mode='same', kernel_initializer='he_normal')(input_layer)
        x11 = BatchNormalization()(x1)
        x11 = PReLU()(x11)

        x2 = Conv1D(filters=out_filter, kernel_size=3, border_mode='same', kernel_initializer='he_normal')(x11)
        #x21 = merge([x1, x2], mode='sum')
        x21 = add([x1, x2])
        x21 = BatchNormalization()(x21)
        x21 = PReLU()(x21)

        x3 = Conv1D(filters=out_filter, kernel_size=3, border_mode='same', kernel_initializer='he_normal')(x21)
        #x31 = merge([x2, x3], mode='sum')
        x31 = add([x2, x3])
        x31 = BatchNormalization()(x31)
        x31 = PReLU()(x31)

        x4 = Conv1D(filters=out_filter, kernel_size=3, border_mode='same', kernel_initializer='he_normal')(x31)
        #x41 = merge([x3, x4], mode='sum')
        x41 = add([x3, x4])
        x41 = BatchNormalization()(x41)
        x41 = PReLU()(x41)

        x5 = Dropout(0.1)(x41)

        return x5

    def build(self, input_shape=((25, 512), (30, 1536))):
        input_text =  Input(shape=input_shape[0])
        input_video =  Input(shape=input_shape[1])

        text = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='text_bilstm-1')(input_text)
        text = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='text_bilstm-2')(text)

        video = Conv1D(filters=self.hidden_size*2, kernel_size=3, border_mode='same', activation='relu')(input_video)

        video = self.recurrent_block(video)
        video = self.recurrent_block(video)

        attention = dot([text, video], axes=[2, 2])
        attention = Activation('softmax', name='softmax')(attention)

        context = dot([attention, video], axes=[2, 1], name='video_context')
        text_combined_context = concatenate([context, text])

        x = self.recurrent_block(text_combined_context)
        x = self.recurrent_block(x)
        #x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'))(text_combined_context)

        x1 = GlobalAveragePooling1D()(x)
        #x2 = GlobalMaxPooling1D()(x)

        #x = concatenate([x1, x2])
        x = Dense(self.hidden_size*2, kernel_initializer='glorot_uniform')(x1)
        x = LeakyReLU(0.2)(x)

        x = Dense(self.no_classes, activation='softmax')(x)

        return Model(inputs=[input_text, input_video], outputs=x)


if __name__ == '__main__':
    model = BiLSTMRecurrentCNN()
    model = model.build()

    model.summary()
