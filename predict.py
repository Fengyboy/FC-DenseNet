import numpy as np
import imp
from PIL import Image
import os
import argparse
import time
import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.utils import floatX
from skimage.io import imsave
# from skimage.transform import resize
from skimage.color import label2rgb


# Default image size
IM_H = 360
IM_W = 480

# Prepare Theano variables for inputs
input_var = T.tensor4('input_var', dtype=theano.config.floatX)


def build_parser():
    parser = argparse.ArgumentParser(description="HED model of NYUD dataset.")

    parser.add_argument(
        "-i", "--input-data-dir",
        help="Path to the root directory of the dataset.",
        dest="input_dir",
        required=True,
    )

    parser.add_argument(
        "-o", "--output-data-dir",
        help="Path to the root directory of the prediction results.",
        dest="output_dir",
        required=True,
    )

    parser.add_argument(
        '-p', '--partition',
        help=("Select partitions to evaluate. Typically train, test and val. "
              "Multiple partitions may be selected. If nothing is passed, all "
              "partitions are evaluated."),
        dest="partition",
        action="append",
        default=None,
    )

    return parser


def read_images(args, image_filenames, image_mean):
    '''
    Take the system args, a list of image file names, and the image_mean array,
    to read and preprocess the list of images.

    Paramters
    ---------
    args: argparse.ArgumentParser()
        The input argements
    image_filenames: list of strings
        The list of input image file names
    image_mean: array_like (1, 3) dtype=float
        The mean of training images, derived from ImageNet training

    Returns
    -------
    im_list: list of array_like (c, h, w) dtype=floatX
        The list of processed images converted to floatX format
    '''
    im_lst = []
    for image in image_filenames:
        image_name = os.path.join(args.input_dir, image + '.png')
        im = Image.open(os.path.join(args.input_dir, image_name))
        rz_im = im.resize((IM_W, IM_H), resample=Image.BICUBIC)
        im_ = np.array(rz_im, dtype=np.float32) / 255
        # Shuffle axes from 01c to c01
        im_ = im_.transpose(2, 0, 1)
        # Convert from RGB to BGR
        im_ = im_[::-1]
        # Subtract mean pixel value
        im_ = im_ - image_mean[:, np.newaxis, np.newaxis]
        im_ = im_.reshape(1, im_.shape[0], im_.shape[1], im_.shape[2])
        im_lst.append(floatX(im_))

    return im_lst


def load_model(config_path, weight_path):
    """
    This function builds the model defined in config_path and restores the weights defined in weight_path. It then
    reports the jaccard and global accuracy metrics on the CamVid test set.
    """

    cf = imp.load_source('cf', config_path)

    ###############
    #  Load data  #
    ###############

    print('-' * 75)
    # Load config file

    ###################
    #  Compile model  #
    ###################

    # Print summary
    net = cf.net
    net.restore(weight_path)

    return net


if __name__ == '__main__':
    args = build_parser().parse_args()
    assert os.path.exists(args.input_dir), (
        "Error, input data directory doesn't exist")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # Load dataset partitioning
    with open(os.path.join(args.input_dir, 'image_list.lst')) as f:
        image_names = f.read().splitlines()

    config_path = 'config/FC-DenseNet103.py'
    weight_path = 'weights/FC-DenseNet103_weights.npz'
    network = load_model(config_path, weight_path)

    # image_mean = np.array((104.00698793, 116.66876762, 122.67891434))
    image_mean = np.array((0., 0., 0.))
    im_lst = read_images(args, image_names, image_mean)

    # Compile test functions
    pred = get_output(network.output_layer, deterministic=True, batch_norm_use_averages=False)
    pred_fn = theano.function([network.input_var], pred)

    for i in range(len(im_lst)):
        start_time = time.time()
        dense_pred = pred_fn(im_lst[i])
        dense_pred = np.argmax(dense_pred, axis=1)
        dense_pred = dense_pred.reshape(IM_H, IM_W)
        print("Prediction of {} took {:.3f}s".format(image_names[i],
                                                     time.time() - start_time))
        max_rgb_name = os.path.join(args.output_dir, image_names[i] + '_rgb.png')
        imsave(max_rgb_name, label2rgb(dense_pred))
