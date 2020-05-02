import keras
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.layers import Input
import keras.backend as K
from scipy.optimize import fmin_l_bfgs_b

IMAGENET_MEAN_RGB_VALUES = [124, 117, 104]
IMAGENET_MEAN_BGR_VALUES = [104, 117, 124]
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25


def preprocess_images(content_image_path, style_image_path):
    content_image = cv2.imread(content_image_path)
    content_image = np.array(content_image).astype(np.float32)
    content_image -= IMAGENET_MEAN_BGR_VALUES

    style_image = cv2.imread(style_image_path)
    style_image = np.array(style_image).astype(np.float32)
    style_image -= IMAGENET_MEAN_BGR_VALUES

    return content_image, style_image


def content_loss(content_image, style_image):
    return K.sum(K.square(content_image - style_image))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return K.sum(K.square(style - combination)) / (4. * (3 ** 2) * (size ** 2))


def total_variation_loss(combination_image):
    a = K.square(combination_image[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - combination_image[:, 1:, :IMAGE_WIDTH-1, :])
    b = K.square(combination_image[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - combination_image[:, :IMAGE_HEIGHT-1, 1:, :])

    return K.sum(K.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))


if __name__ == "__main__":
    content_image, style_image = preprocess_images("./content_image.png", "./style_image.png")

    input_image = K.variable(np.array([content_image]))
    style_image = K.variable(np.array([style_image]))
    combination_image = K.placeholder((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    input_tensor = K.concatenate([input_image, style_image, combination_image], axis=0)
    vgg16 = VGG16(input_tensor=input_tensor, include_top=False)
    #vgg16.predict(np.array([content_image, content_image, content_image]))

    loss = K.variable(0.)

    layers = dict([(layer.name, layer.output) for layer in vgg16.layers])

    # Calc content weight
    content_layer = "block2_conv2"
    layer_features = layers[content_layer]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    print(content_loss(content_image_features, combination_features))
    print(loss)
    loss += CONTENT_WEIGHT * content_loss(content_image_features, combination_features)
    print(loss)

    # Calc style weight
    style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    for layer_name in style_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss = compute_style_loss(style_features, combination_features)
        loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

    # Calc total loss
    loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

    outputs = [loss]
    outputs += K.gradients(loss, combination_image)


    def evaluate_loss_and_gradients(x):
        x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        outs = K.function([combination_image], outputs)([x])
        loss = outs[0]
        gradients = outs[1].flatten().astype("float64")
        return loss, gradients


    class Evaluator:

        def loss(self, x):
            loss, gradients = evaluate_loss_and_gradients(x)
            self._gradients = gradients
            return loss

        def gradients(self, x):
            return self._gradients


    evaluator = Evaluator()

    x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

    for i in range(1):
        x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
        print("Iteration %d completed with loss %d" % (i, loss))

    x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
    x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
    x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
    x = np.clip(x, 0, 255).astype("uint8")
    output_image = Image.fromarray(x)
    plt.imshow(output_image)
    plt.show()

