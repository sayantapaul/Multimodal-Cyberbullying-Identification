from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, dot, concatenate, Activation
from keras.layers import merge, Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization, GlobalAveragePooling1D
from keras.layers.advanced_activations import PReLU

class RecurrentCNN:
    def __init__(self, no_filters, no_classes=2):
        self.no_filters = no_filters
        self.no_classes = no_classes

    def recurrent_block(self, input_layer):
        out_filter = self.no_filters

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
        text_input = Input(shape=input_shape[0])
        video_input = Input(shape=input_shape[1])

        text = Convolution1D(filters=self.no_filters, kernel_size=3, border_mode='same', activation='relu')(text_input)
        video = Convolution1D(filters=self.no_filters, kernel_size=3, border_mode='same', activation='relu')(video_input)

        text = self.recurrent_block(text)
        video = self.recurrent_block(video)

        attention = dot([text, video], axes=[2, 2])
        attention = Activation('softmax', name='softmax')(attention)

        context = dot([attention, video], axes=[2, 1], name='video_context')
        text_combined_context = concatenate([context, text])

        x = GlobalAveragePooling1D()(text_combined_context)
        x = Dense(self.no_classes, activation='softmax')(x)

        return Model(inputs=[text_input, video_input], outputs=x)


if __name__ == '__main__':
    model = RecurrentCNN(128)
    model = model.build()

    model.summary()
