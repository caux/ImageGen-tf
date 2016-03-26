import argparse
import numpy as np
import NeuralNet
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

    data = NeuralNet.initializeData(imgSettings.width, imgSettings.height)

    # Output Image
    outputImage = np.empty(imgSettings.width * imgSettings.height * 3, dtype=np.uint8)

    input, layers, weights = NeuralNet.buildNeuralNet(netSettings.input, netSettings.hidden, netSettings.output)

    result = NeuralNet.runNeuralNet(input, layers[-1], data)

    # transform result in image data
    for index in range(len(data)):
        outputImage[index * 3] = int(result[index][imgSettings.mapR] * imgSettings.maxR)
        outputImage[index * 3 + 1] = int(result[index][imgSettings.mapG] * imgSettings.maxG)
        outputImage[index * 3 + 2] = int(result[index][imgSettings.mapB] * imgSettings.maxB)

    with open(args.filename, 'w') as outfile:
        pngWriter = png.Writer(imgSettings.width, imgSettings.height)
        pngWriter.write(outfile, np.reshape(outputImage, (-1, imgSettings.width*3)))

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
