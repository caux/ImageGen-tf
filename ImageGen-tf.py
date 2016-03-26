import argparse
import numpy as np
import tensorflow as tf
import png
import time

from collections import namedtuple

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Output png.")
    parser.add_argument("-i", help="Number of input nodes.", type=int, default=16)
    parser.add_argument("-l", help="Number of hidden layers.", type=int, default=12)
    parser.add_argument("-x", help="Image width.", type=int, default=800)
    parser.add_argument("-y", help="Image height.", type=int, default=600)

    args = parser.parse_args()

    # Initialize settings
    NetSettings = namedtuple("NetSettings", "input hidden output")
    ImgSettings = namedtuple("ImgSettings", "width height "
                                            "mapR maxR "
                                            "mapG maxG "
                                            "mapB maxB")

    netSettings = NetSettings(args.i, args.l, 3)
    imgSettings = ImgSettings(args.x, args.y,
                              2, 155,
                              2, 155,
                              2, 155)

    # Input Vector
    data = np.empty([imgSettings.width * imgSettings.height, 3])

    for y in range(imgSettings.height):
        for x in range(imgSettings.width):
            index = y * imgSettings.width + x
            scaledX = float(x) / imgSettings.width
            scaledY = float(y) / imgSettings.height
            data[index][0] = scaledX
            data[index][1] = scaledY
            data[index][2] = 1.0  # bias

    x = tf.placeholder(tf.float32, shape=[None, 3])

    # Output Image
    outputImage = np.empty(imgSettings.width * imgSettings.height * 3, dtype=np.uint8)

    layers = []
    weights = []

    # Input Node
    weightsInput = tf.Variable(
        tf.random_normal([3, netSettings.input], name="InputWeights"))
    layerInput = tf.tanh(tf.matmul(x, weightsInput))

    weights.append(weightsInput)
    layers.append(layerInput)

    # Hidden Node
    for layer in range(1, netSettings.hidden + 1):
        name = "HiddenWeights" + str(layer)
        weightsHidden = tf.Variable(tf.random_normal([netSettings.input,
                                                      netSettings.input],
                                                     name=name))
        layerHidden = tf.tanh(tf.matmul(layers[-1], weightsHidden))

        weights.append(weightsHidden)
        layers.append(layerHidden)

    # Output Node
    weightsOutput = tf.Variable(
        tf.random_normal([netSettings.input, netSettings.output],
                         name="OutputWeights"))
    y = tf.sigmoid(tf.matmul(layers[-1], weightsOutput))

    weights.append(weightsOutput)
    layers.append(y)

    # Run
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        result = sess.run(y, feed_dict={x: data})

        for index in range(len(data)):

            r = int(result[index][imgSettings.mapR] * imgSettings.maxR)
            g = int(result[index][imgSettings.mapG] * imgSettings.maxG)
            b = int(result[index][imgSettings.mapB] * imgSettings.maxB)

            outputImage[index * 3] = r
            outputImage[index * 3 + 1] = g
            outputImage[index * 3 + 2] = b

    with open(args.filename, 'w') as outfile:
        pngWriter = png.Writer(imgSettings.width, imgSettings.height)
        pngWriter.write(outfile, np.reshape(outputImage, (-1, imgSettings.width*3)))

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
