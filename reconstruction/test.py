"""Tests the model"""

import argparse
import logging
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pickle

import data_loader
import torch
from scipy.io import savemat
from torch import nn

import net
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=25, help="Index of example instance for testing")
parser.add_argument('--test_data_dir', default=os.path.dirname(os.path.abspath(__file__)) + '/../data_30/test_data.mat', help="Path containing the testing dataset")
parser.add_argument('--model_dir', default=os.path.dirname(os.path.abspath(__file__)) +'/logs', help="Directory containing the model")


def plotTensor( tensor ):

    outputs = tensor.data
    outputs=outputs.cpu()
    np_outputs=outputs.numpy()
    np_outputs.astype(float)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(np_outputs > 0.5, facecolors='red', edgecolor='k')
    plt.show()

def test_all(model, test_data, test_labels, tesize):
    """
    Takes all the test samples and outputs the completion and the average denoising error
    """

    # set model to evaluation mode
    model.eval()

    test_data.cuda()

    inputs = torch.Tensor(tesize, 1, 30, 30, 30).cuda()
    outputs = torch.Tensor(tesize, 30, 30, 30).cuda()
    perfect_cubes = torch.Tensor(tesize, 1, 30, 30, 30).cuda()

    for k in range(tesize):
        input = test_data[k]
        input=input.reshape(1,1,30,30,30)
        input = input.float()
        input = input.cuda()

        perfect_cubes[k] = test_labels[k].double()
        
        inputs[k] = input
        output=model.forward(input)
        outputs[k]=output.detach()

    # Calculation of Dice Coefficients as error estimate:

    bin_output = torch.gt(outputs, 0.5)  # gt means >
    bin_output = bin_output.double()

    m1 = bin_output.view(bin_output.size(0), -1)  # Flatten
    m2 = perfect_cubes.view(perfect_cubes.size(0), -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection) / (m1.sum() + m2.sum())


def test_instance(model, i, test_data, test_labels):
    """
    this function is meant to feed the corrupted 3D test data to the network and save the output in binary format
    -- this output can then be read and visualized in matlab.
    -- computes the reconstruction error and the BCE loss for this instance
    """

    # set model to evaluation mode
    model.eval()

    #error = reconstructed - original input
    err = 0

    test_labels = test_labels.contiguous()

    inputs=torch.Tensor(1, 1, 30, 30, 30)
    inputs=inputs.cuda()

    input=test_data[i]
    input=input.cuda()
    inputs[0]=input

    perfect_cube=test_labels[i]
    outputs=model.forward(inputs)
    outputs=outputs.cuda()
    perfect_cube = perfect_cube.cuda()

    bin_output = torch.gt(outputs, 0.5)  # gt means >
    bin_output = bin_output.double()

    # Calculate Dice Similarity
    m1 = bin_output.view(-1)  # Flatten
    m2 = perfect_cube.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    err = (2. * intersection) / (m1.sum() + m2.sum())

    # Calculate BCE-Loss
    perfect_cube=perfect_cube.float()
    loss_fn = nn.BCELoss()
    bce=loss_fn( outputs.squeeze(), perfect_cube)

    return err, bce, outputs


def save_output(outputs, filename):

    """
    Saves the reconstruction output in a .mat file
    """

    outputs=outputs.reshape(30, 30, 30)
    outputs=torch.squeeze(outputs)

    dims=outputs.ndimension()

    if dims > 1:
        for i in range(math.floor(dims/2)-1):
            outputs=torch.transpose(outputs, i, dims-i-1)
        outputs=outputs.contiguous()

    outputs=outputs.data
    outputs=outputs.cpu()
    np_outputs=outputs.numpy()
    np_outputs.astype(float)
    savemat(filename, dict([('output', np_outputs)]))


if __name__ == '__main__':

    args = parser.parse_args()
    i=int( args.i )

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    #Load testing data
    logging.info("Loading the test dataset...")
    test_data, tesize = data_loader.load_data(args.test_data_dir, 'dataset')
    logging.info("Number of testing examples: {}".format(tesize))

    test_labels, tesize_labels = data_loader.load_data(args.test_data_dir, 'labels')

    #initialize the model
    autoencoder = net.VolAutoEncoder()
    autoencoder.cuda()

    #reload weights of the trained model from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, 'last.pth.tar'), autoencoder)

    # ------test all-----------
    te_err = test_all(autoencoder, test_data, test_labels, tesize)
    # logging.info("Test All: The test error is {} %".format(te_err))
    logging.info("Average Dice Similarity is {}".format(te_err))

    # ------test instance-----------w/ Recons Error and BCE loss
    te_err, bce, outputs = test_instance(autoencoder, i, test_data, test_labels)
    logging.info("Test instance {}: Dice Similarity is {} and BCE Loss is {}".format(i, te_err,bce))

    dice_network = []
    for index in range( tesize ):
        diceScore, bce, outputs = test_instance(autoencoder, index,test_data, test_labels)
        dice_network.append( diceScore.item() )

    with open("reconstructionDice.txt", "wb") as fp:   #Pickling
        pickle.dump(dice_network, fp)

    fig, ax = plt.subplots()
    plt.gca().set_title('Reconstruction Dice-Scores')
    bplt = plt.boxplot( [dice_network], patch_artist=True )
    bplt['boxes'][0].set( facecolor = 'red' )
    plt.legend([bplt["boxes"][0]], ['Network DCS'], loc='upper right')
    plt.show()