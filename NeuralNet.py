import tensorflow as tf
import numpy as np
import random

def initializeData(width, height):
        # Input Vector
    data = np.empty([width * height, 3])

    for y in range(height):
        for x in range(width):
            index = y * width + x
            scaledX = float(x) / width
            scaledY = float(y) / height
            data[index][0] = scaledX
            data[index][1] = scaledY
            data[index][2] = 1.0  # bias

    return data

def buildNeuralNet(inputNodes, hiddenLayers, outputNodes):
    layers = []
    weights = []

    x = tf.placeholder(tf.float32, shape=[None, 3])

    # Input Layer
    weightsInput = tf.Variable(
        tf.random_normal([3, inputNodes], name="InputWeights"))
    layerInput = tf.tanh(tf.matmul(x, weightsInput))

    weights.append(weightsInput)
    layers.append(layerInput)

    # Hidden Layer
    for layer in range(1, hiddenLayers + 1):
        name = "HiddenWeights" + str(layer)
        weightsHidden = tf.Variable(tf.random_normal([inputNodes,
                                                      inputNodes],
                                                     name=name))
        layerHidden = tf.tanh(tf.matmul(layers[-1], weightsHidden))

        weights.append(weightsHidden)
        layers.append(layerHidden)

    # Output Layer
    weightsOutput = tf.Variable(
        tf.random_normal([inputNodes, outputNodes],
                         name="OutputWeights"))
    y = tf.sigmoid(tf.matmul(layers[-1], weightsOutput))

    weights.append(weightsOutput)
    layers.append(y)

    return x, layers, weights

def runNeuralNet(input, node, data):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        result = sess.run(node, feed_dict={input: data})

    return result