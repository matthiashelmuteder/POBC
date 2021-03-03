"""Tests the model"""

import argparse
import logging
import math
import os
import nrrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import pickle

import pandas as pd

import data_loader
import torch
from scipy.io import savemat
from scipy import ndimage
from torch import nn

import net
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=97, help="Index of example instance for testing")
parser.add_argument('--test_data_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/../data_30/test_data.mat', help="Path to the training dataset")
parser.add_argument('--target_data_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/../data_60/test_data.mat', help="Path to the target dataset")
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

def calcDiceSim( modelA, modelB):
    # Calculate Dice Similarity
    m1 = modelA.cuda().double().view(-1)  # Flatten
    m2 = modelB.cuda().double().view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    err = (2. * intersection) / (m1.sum() + m2.sum())
    return err

def test_all(model, test_data, test_labels, tesize):
    """
    Takes all the test samples and outputs the completion and the average denoising error
    """

    # set model to evaluation mode
    model.eval()

    test_data.cuda()

    inputs = torch.Tensor(tesize, 1, 30, 30, 30).cuda()
    outputs = torch.Tensor(tesize, 1, 60, 60, 60).cuda()
    perfect_cubes = torch.Tensor(tesize, 1, 60, 60, 60).cuda()

    for k in range(tesize):
        input = test_data[k]
        input=input.reshape(1, 1, 30, 30, 30 )
        input = input.float()
        input=input.cuda()

        perfect_input = test_labels[k]
        perfect_input = perfect_input.double()

        perfect_cubes[k] = perfect_input
        inputs[k] = input
        output=model.forward(input)
        outputs[k]=output.detach()

    bin_output = torch.gt(outputs, 0.5)  # gt() means >
    bin_output = bin_output.double()

    m1 = bin_output.view(bin_output.size(0), -1)  # Flatten
    m2 = perfect_cubes.view(perfect_cubes.size(0), -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection) / (m1.sum() + m2.sum())


def test_instance( model, i, test_data, test_labels ):
    """
    this function is meant to feed the corrupted 3D test data to the network and save the output in binary format
    -- this output can then be read and visualized in matlab.
    -- computes the reconstruction error and the BCE loss for this instance
    """

    # set model to evaluation mode
    model.eval()

    inputs=torch.Tensor(1,1,30,30,30)
    inputs=inputs.cuda()

    input=test_data[i]
    input=input.cuda()
    inputs[0]=input

    perfect_cube=test_labels[i]
    outputs=model.forward(inputs)
    outputs=outputs.float()

    outputs=outputs.cuda()
    perfect_cube = perfect_cube.cuda()
    
    bin_output = torch.gt(outputs, 0.5)  # gt means >
    bin_output = bin_output.double()
    
    # Calculate Dice Similarity
    m1 = bin_output.view(-1)  # Flatten
    m2 = perfect_cube.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection) / (m1.sum() + m2.sum())

    # Calculate BCE-Loss
    perfect_cube=perfect_cube.float()
    loss_fn = nn.BCELoss()
    bce=loss_fn( outputs.squeeze(), perfect_cube )
    
    return dice, bce, outputs

def test_instance_cubic( i, test_data, test_labels ):

    loSkull = test_data[i].squeeze().cpu().numpy()
    squUpsampled = ndimage.zoom( loSkull, 2.2, order=2 ) > 0.70

    # Crop the linear upsampled skulls since we had to zoom by 2.2 instead of only 2 to match the skull sizes
    l = squUpsampled.shape[0]
    squUpsampled = squUpsampled[ int(l/2)-30:int(l/2)+30, int(l/2)-30:int(l/2)+30, int(l/2)-30:int(l/2)+30 ]
    dice_cubicSkull = calcDiceSim( torch.from_numpy( squUpsampled ), test_labels[i] ).item()

    return dice_cubicSkull, squUpsampled


def save_output(outputs, filename):

    """
    Saves the reconstruction output in a .mat file
    """

    # outputs=outputs.reshape(30,30,30)
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
    i=args.i
    i=int(i)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    #Load testing data
    logging.info("Loading the test dataset...")
    test_data, tesize = data_loader.load_data(args.test_data_dir, 'labels')
    logging.info("Number of testing examples: {}".format(tesize))

    test_labels, tesize_labels = data_loader.load_data(args.target_data_dir, 'labels')

    #initialize the model
    autoencoder = net.VSSR()
    autoencoder.cuda()

    #reload weights of the trained model from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, 'last.pth.tar'), autoencoder)


    #------test all-----------
    # te_err = test_all(autoencoder, test_data, test_labels, tesize)
    # logging.info("Average Dice Similarity is {}".format(te_err))

        
    #------test instance-----------w/ Recons Error and BCE loss
    diceScore, bce, outputs=test_instance(autoencoder,i,test_data,test_labels)
    logging.info("Test instance {}: Dice Similarity is {} and BCE Loss is {}".format(i, diceScore, bce))

    outputs=outputs.squeeze().cpu()
    np_outputs=outputs.detach().numpy()
    np_outputs.astype(float)
    nrrd.write( 'output.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

    outputs = test_data[i]
    outputs = outputs.cpu()
    np_outputs = outputs.numpy()
    nrrd.write( 'input.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

    dice_network = []
    dice_interpolation = []
    for index in range( tesize ):
        diceScore, bce, outputs = test_instance(autoencoder, index,test_data, test_labels)
        dice_network.append( diceScore.item() )

        diceScore, outputs = test_instance_cubic(index, test_data, test_labels)
        dice_interpolation.append( diceScore )

    with open("superResolutionDice.txt", "wb") as fp:   #Pickling
       pickle.dump([dice_network, dice_interpolation], fp)

    boxcolour=['red', 'blue']
    fig, ax = plt.subplots()
    plt.gca().set_title('Super-Resolution Dice-Scores')
    bplt = plt.boxplot( [dice_network, dice_interpolation], patch_artist=True )
    bplt['boxes'][0].set( facecolor = boxcolour[0] )
    bplt['boxes'][1].set( facecolor = boxcolour[1] )
    plt.legend([bplt["boxes"][0], bplt["boxes"][1]], ['Network DCS', 'Interpolation DCS'], loc='upper right')
    plt.show()
    
    # logging.info("Highest Dice Similarity is {} and BCE Loss is {}".format( highScore, loBCE ))
    # logging.info("Lowest Dice Similarity is {} and BCE Loss is {}".format( loScore, highBCE ))
    # logging.info("Average Dice Similarity is {}".format( averageScore / tesize ))
    # logging.info("Index: {}".format( loIndex ))


    
    
