import argparse
import os.path
import copy

import torch
import nrrd
import numpy
import pickle
import cc3d
from scipy import ndimage

import data_loader
import utils
import trained_networks.reconstruction.net as rec
import trained_networks.superresolution.net as vsr
import pandas as pd
import matplotlib.pyplot as plt

home = os.path.dirname(os.path.abspath(__file__)) + "/nrrd"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reconstruction_model',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + '/trained_networks/reconstruction/',
        nargs='?',
        help='Path to trained reconstruction model.'
    )
    parser.add_argument(
        '--superresolution_model',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + '/trained_networks/superresolution/',
        nargs='?',
        help='Path to trained superresolution model.'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=49,
        nargs='?',
        help='Index of used data inside the datasets'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + '/../data_30/test_data.mat',
        help="Path to the training dataset")
    parser.add_argument(
        '--target_data_dir',
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + '/../data_60/test_data.mat', 
        help="Path to the target dataset"
    )
    parser.add_argument(
        '--noOutput',
        action='store_true',
        default=False,
        help='Do not write nrrd output files'
    )
    parser.add_argument(
        '--testAll',
        action='store_true',
        default=True,
        help='Calculate dice scores for the whole dataset'
    )

    return parser.parse_known_args()

def calcDiceSim( modelA, modelB):
    # Calculate Dice Similarity
    m1 = modelA.cuda().double().view(-1)  # Flatten
    m2 = modelB.cuda().double().view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    err = (2. * intersection) / (m1.sum() + m2.sum())
    return err

def filterSkull( input, close=False, open=False ):
    # Filter the difference to get rid of network halucinations
    data = copy.deepcopy( input )
    if( close ):
        data = ndimage.binary_closing(data)
    if( open ):
        data = ndimage.binary_opening(data)

    # connected components to get biggest chunk (hopefully the implant)
    components = cc3d.connected_components( data, connectivity=6 )
    numLabels = numpy.max( components )
    biggest = 0
    biggestId = 0
    for segid in range(1, numLabels+1):
        comp = components * ( components == segid )
        compSize = comp.sum() / segid 
        if( compSize > biggest ):
            biggest = compSize
            biggestId = segid

    components = components == biggestId 
    return input * components

def calcNetworkOutputs( loDefect, highDefect ):

    with torch.no_grad():
        input = loDefect.reshape(1, 1, 30, 30, 30).float().cuda()
        defSkull = highDefect.reshape(1, 1, 60, 60, 60).float().cuda()
        
        # Put input through networks and threshold it
        reconstructed = reconstructor.forward(input)
        voxel_reconstructed = torch.gt( reconstructed, 0.5 ).float().cuda()
        upsampled = upsampler.forward( voxel_reconstructed )
        voxel_upsampled = torch.gt( upsampled, 0.5 ).float().cuda()

        # generate implant from upsampled skull and filter it
        highDiff = ( voxel_upsampled - defSkull ).gt( 0 ).squeeze().cpu().numpy()
        implant = filterSkull( highDiff )

        return voxel_reconstructed, voxel_upsampled, highDiff, implant

def calcInterpolationOutput( reconstructed, highDefect ):

    defSkull = highDefect.reshape(1, 1, 60, 60, 60).float().cuda()

    # Calculate direct interpolations to compare them to the network output
    reconstucted_np = reconstructed.squeeze().cpu().numpy()
    squUpsampled = ndimage.zoom( reconstucted_np, 2.2, order=2 ) > 0.70

    # Crop the upsampled skulls since we had to zoom by 2.2 instead of only 2 to match the skull sizes
    l = squUpsampled.shape[0]
    squUpsampled = squUpsampled[ int(l/2)-30:int(l/2)+30, int(l/2)-30:int(l/2)+30, int(l/2)-30:int(l/2)+30 ]

    squDiff = (( squUpsampled - defSkull.squeeze().cpu().numpy() ) > 0 )
    squImplant = filterSkull( squDiff, open=True )

    return squUpsampled, squDiff, squImplant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unparsed = parse_arguments()

    # initialize reconstruction model and load saved values
    reconstructor = rec.VolAutoEncoder().cuda().eval()
    utils.load_checkpoint(os.path.join(
        args.reconstruction_model, 'last.pth.tar'), reconstructor)

    # initialize superresolution model
    upsampler = vsr.VSSR().cuda().eval()
    utils.load_checkpoint(os.path.join(
        args.superresolution_model, 'last.pth.tar'), upsampler)

    # load datasets
    loDefect, tesize = data_loader.load_data( args.test_data_dir, 'dataset' )
    loFull, resize = data_loader.load_data( args.test_data_dir, 'labels' )
    highDefect, dsize = data_loader.load_data( args.target_data_dir, 'dataset' )
    highFull, lsize = data_loader.load_data( args.target_data_dir, 'labels' )

    if( args.testAll ):

        networkSkullScores = []
        networkImpScores = []
        unfilteredNetImpScore = []

        interpolationSkullScores = []
        interpolationImpScores = []

        for i in range( tesize ):

            voxel_reconstructed, voxel_upsampled, highDiff, implant = calcNetworkOutputs( loDefect[i], highDefect[i] )
            squUpsampled, squDiff, squImplant = calcInterpolationOutput( voxel_reconstructed, highDefect[i])

            targetImplant = ( highFull[i] - highDefect[i] ).gt( 0 ).squeeze().cpu().numpy()
            unfilteredNetImpScore.append( calcDiceSim( torch.from_numpy( highDiff ), torch.from_numpy( targetImplant ) ).item() )

            networkSkullScores.append( calcDiceSim( voxel_upsampled, highFull[i] ).item() )
            networkImpScores.append( calcDiceSim( torch.from_numpy( implant ), torch.from_numpy( targetImplant ) ).item() )

            interpolationSkullScores.append( calcDiceSim( torch.from_numpy(squUpsampled), highFull[i] ).item() )
            interpolationImpScores.append( calcDiceSim( torch.from_numpy(squImplant), torch.from_numpy( targetImplant ) ).item() )


            print( "Calculate implant:" + str(i) )

        with open("implantDice.txt", "wb") as fp:   #Pickling
            pickle.dump([networkImpScores, interpolationImpScores, networkSkullScores, interpolationSkullScores], fp)

        boxcolour=['red', 'blue']
        fig, ax = plt.subplots()
        plt.gca().set_title('Implant Generation Scores')
        bplt = plt.boxplot( [networkImpScores, interpolationImpScores], patch_artist=True )
        bplt['boxes'][0].set( facecolor = boxcolour[0] )
        bplt['boxes'][1].set( facecolor = boxcolour[1] )
        plt.legend([bplt["boxes"][0], bplt["boxes"][1]], ['Network DCS', 'Interpolation DCS'], loc='upper right')
        plt.show()

        fig, ax = plt.subplots()
        plt.gca().set_title('Super-Resolution Scores')
        bplt = plt.boxplot( [networkSkullScores, interpolationSkullScores], patch_artist=True )
        bplt['boxes'][0].set( facecolor = boxcolour[0] )
        bplt['boxes'][1].set( facecolor = boxcolour[1] )
        plt.legend([bplt["boxes"][0], bplt["boxes"][1]], ['Network DCS', 'Interpolation DCS'], loc='upper right')
        plt.show()     

        zerosNet = len( [ score for score in networkImpScores if score == 0.0 ] )
        zerosImp = len( [ score for score in interpolationImpScores if score == 0.0 ] )
        networkImpScores = [ score for score in networkImpScores if score != 0.0 ]
        interpolationImpScores = [ score for score in interpolationImpScores if score != 0.0 ]
        
        print("Average Network Skull Score: " + str( sum( networkSkullScores ) / len( networkSkullScores ) ) + "\n" +
            "Min. Network Skull Score: " + str( min( networkSkullScores ) ) + "\n" +
            "Max. Network Skull Score: " + str( max( networkSkullScores ) ) + "\n" +
            "Average Network Implant Score: " + str( sum( networkImpScores ) / len( networkImpScores ) ) + "\n" +
            "Min. Network Implant Score: " + str( min( networkImpScores ) ) + "\n" +
            "Max. Network Implant Score: " + str( max( networkImpScores ) ) + "\n" +
            "Number of failed network implants: " + str( zerosNet ) + "\n\n" +

            "Average Interpolated Skull Score: " + str( sum( interpolationSkullScores ) / len( interpolationSkullScores ) ) + "\n" +
            "Min. Interpolated Skull Score: " + str( min( interpolationSkullScores ) ) + "\n" +
            "Max. Interpolated Skull Score: " + str( max( interpolationSkullScores ) ) + "\n" +
            "Average Interpolated Implant Score: " + str( sum( interpolationImpScores ) / len( interpolationImpScores ) ) + "\n" +
            "Min. Interpolated Implant Score: " + str( min( interpolationImpScores ) ) + "\n" +
            "Max. Interpolated Implant Score: " + str( max( interpolationImpScores ) ) + "\n" +
            "Number of failed interpolated implants: " + str( zerosImp ) + "\n" )


    # Calculate outputs for index arg.index
    targetImplant = ( highFull[args.index] - highDefect[args.index] ).gt( 0 ).squeeze().cpu().numpy()
    voxel_reconstructed, voxel_upsampled, highDiff, implant = calcNetworkOutputs( loDefect[args.index], highDefect[args.index] )
    squUpsampled, squDiff, squImplant = calcInterpolationOutput( voxel_reconstructed, highDefect[args.index] )


    # Save output as nrrd to inspect them with programs like itk-snap
    if( not args.noOutput ):

        voxel_reconstructed, voxel_upsampled, highDiff, implant = calcNetworkOutputs( loDefect[args.index], highDefect[args.index] )
        squUpsampled, squDiff, squImplant = calcInterpolationOutput( voxel_reconstructed, highDefect[args.index])

        if not os.path.exists( home ):
            os.mkdir( home )

        outputs = torch.squeeze(voxel_reconstructed)
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/reconstructed.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        outputs = torch.squeeze(voxel_upsampled)
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/output.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        outputs = highFull[args.index]
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/target.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        outputs = loDefect[args.index]
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/input.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        outputs = loFull[args.index]
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/recTarget.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        outputs = highDefect[args.index]
        outputs = outputs.cpu()
        np_outputs = outputs.numpy()
        nrrd.write( home + '/defect.nrrd', numpy.array(np_outputs > 0.5).astype(float) )

        nrrd.write( home + '/implant.nrrd', implant.astype(float) )
        nrrd.write( home + '/squImplant.nrrd', squImplant.astype(float) )
        nrrd.write( home + '/targetImplant.nrrd', targetImplant.astype(float) )

        nrrd.write( home + '/unfiltered.nrrd', highDiff.astype(float) )
        nrrd.write( home + '/squUpsampled.nrrd', squUpsampled.astype(float))


