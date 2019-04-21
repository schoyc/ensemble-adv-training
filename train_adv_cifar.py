import keras
from keras import backend as K

import tensorflow as tf

from tensorflow.python.platform import flags
from keras.models import save_model, load_model

from train_cifar10_resnet import *
from tf_utils import tf_train, tf_test_error_rate
from attack_utils import gen_grad
from fgs import symbolic_fgs

FLAGS = flags.FLAGS

def set_cifar_flags():
    try:
        flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of training batches')
    except argparse.ArgumentError:
        pass

    flags.DEFINE_integer('NUM_CLASSES', 10, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')


def get_cifar_data():
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def data_gen_cifar(x_train):
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

def load_cifar_model(weights_path):
    model = keras.models.load_model(weights_path)
    logits = Model(inputs=model.input, outputs=model.layers[-2].output)
    return logits


def main(model_name, eat_model_name, adv_model_names, train_ensemble):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    # set_cifar_flags()

    # flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = get_cifar_data()

    data_gen = data_gen_cifar(X_train)

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------


    if train_ensemble:
        # ResNet44 v1
        _, model1_name = train_resnet(version=1, n=7)
        # ResNet56 v2
        _, model2_name = train_resnet(version=2, n=6)
        # ResNet110 v2
        _, model3_name = train_resnet(version=2, n=12)
        adv_model_names = [model1_name, model2_name, model3_name]

    adv_models = [None] * len(adv_model_names)
    for i in range(len(adv_model_names)):
        adv_models[i] = load_cifar_model(adv_model_names[i])

    x = tf.placeholder(tf.float32, shape=(None,
                             32,
                             32,
                             3))

    y = tf.placeholder(tf.float32, shape=(None, 10))

    eps = args.eps

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models

    model = load_cifar_model(model_name)

    x_advs = [None] * (len(adv_models) + 1)

    for i, m in enumerate(adv_models + [model]):
        logits = m(x)
        grad = gen_grad(x, logits, y, loss='training')
        x_advs[i] = symbolic_fgs(x, grad, eps=eps)

    # Train an MNIST model
    tf_train(x, y, model, X_train, Y_train, data_gen, x_advs=x_advs, epochs=args.epochs)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)

    model_name = eat_model_name
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'wr') as f:
        f.write(json_string)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model")
    parser.add_argument("--eat_model", help="path to save EAT model")
    parser.add_argument('--adv_models', nargs='*',
                        help='path to adv model(s)')
    parser.add_argument("--type", type=int, help="model type", default=0)
    parser.add_argument("--epochs", type=int, default=12,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--train-ensemble", action='store_true')

    args = parser.parse_args()
    main(args.model, args.eat_model, args.adv_models, args.train_ensemble)
