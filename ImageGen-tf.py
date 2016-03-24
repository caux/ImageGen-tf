import tensorflow as tf
from collections import namedtuple

# Initialize settings
NetSettings = namedtuple("NetSettings", "input hidden output")
ImgSettings = namedtuple("ImgSettings", "width height mapR maxR mapG maxG mapB maxB")

netSettings = NetSettings(22, 14, 3)
imgSettings = ImgSettings(800, 600,
                          2, 155,
                          2, 155,
                          2, 155)

# Data

# Input Vector
x = tf.placeholder(tf.float32, [1, 3])

layers = []
weights = []

# Input Node
weightsInput = tf.Variable(tf.random_normal([3, netSettings.input], name="InputWeights"))
layerInput = tf.tanh(tf.matmul(x, weightsInput))

layers.append(layerInput)
weights.append(weightsInput)

# Hidden Node
for layer in range(1, netSettings.hidden + 1):
    name = "HiddenWeights" + str(layer)
    weightsHidden = tf.Variable(tf.random_normal([netSettings.input, netSettings.input],
                                                 name=name))
    layerHidden = tf.tanh(tf.matmul(layers[layer - 1], weightsHidden))

    layers.append(weightsHidden)
    weights.append(layerHidden)

# Output Node
weightsOutput = tf.Variable(
    tf.random_normal([netSettings.input, netSettings.output], name="OutputWeights"))
layerInput = tf.sigmoid(tf.matmul(layers[-1], weightsOutput))

layers.append(layerInput)
weights.append(weightsInput)

# Initialize Variables
init = tf.initialize_all_variables()
with tf.Session() as sess:
    result = sess.run(init)
    result = sess.run([layers[-1]])
    print(result)
