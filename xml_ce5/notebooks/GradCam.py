import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
import cv2

image = np.array(load_img("../data/cat.jpg", target_size=(224, 224, 3)))
plt.imshow(image)

model = ResNet50()

last_conv_layer = model.get_layer("conv5_block3_out")
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ["avg_pool", "predictions"]:
    x = model.get_layer(layer_name)(x)

classifier_model = tf.keras.Model(classifier_input, x)


multiobject_image = np.array(
    load_img("../data/cat_and_dog.jpg", target_size=(224, 224, 3))
)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(multiobject_image[np.newaxis, ...])
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(-1 * grads, axis=(0, 1, 2))

last_conv_layer_output = last_conv_layer_output.numpy()[0]
pooled_grads = pooled_grads.numpy()
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# Average over all the filters to get a single 2D array
ctfcl_gradcam = np.mean(last_conv_layer_output, axis=-1)
# Normalise the values
ctfcl_gradcam = np.clip(ctfcl_gradcam, 0, np.max(ctfcl_gradcam)) / np.max(ctfcl_gradcam)
ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (224, 224))

plt.imshow(multiobject_image)
plt.imshow(ctfcl_gradcam, alpha=0.5)

mask = cv2.resize(ctfcl_gradcam, (224, 224))
mask[mask > 0.1] = 255
mask[mask != 255] = 0
mask = mask.astype(bool)

ctfctl_image = multiobject_image.copy()
ctfctl_image[mask] = (0, 0, 0)

plt.imshow(ctfctl_image)

decode_predictions(model.predict(image[np.newaxis, ...]))
[[('n02127052', 'lynx', 0.3293164),
  ('n02123045', 'tabby', 0.18094611),
  ('n02123159', 'tiger_cat', 0.1474288),
  ('n02124075', 'Egyptian_cat', 0.09377903),
  ('n02128757', 'snow_leopard', 0.0360316)]]


decode_predictions(model.predict(ctfctl_image[np.newaxis, ...]))
[[('n02124075', 'Egyptian_cat', 0.60261524),
  ('n02123045', 'tabby', 0.040005833),
  ('n04254777', 'sock', 0.036930993),
  ('n02834397', 'bib', 0.03177831),
  ('n03814639', 'neck_brace', 0.017811844)]]