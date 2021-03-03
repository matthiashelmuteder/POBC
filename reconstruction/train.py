import argparse
import logging
import os

import data_loader
import torch
import torch.optim as optim
from tqdm import tqdm

import net
import test
import utils

#argument parser
parser=argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/../data_30/train_data.mat', help="Path to the training dataset")
parser.add_argument('--test_data_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/../data_30/test_data.mat', help="Path to the test dataset")
parser.add_argument('--model_dir', default=os.path.dirname(os.path.abspath(__file__)) +'/logs', help="Directory containing the model")
parser.add_argument('--restore_file', default=os.path.dirname(os.path.abspath(__file__)) +'/logs/last',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")

#hyperparameters
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)


def train(model, optimizer, loss_fn, train_data, train_label, trsize, batch_size): #lets see what else
    """
    :param model: the neural network (autoencoder)
    """

    #set model to training mode
    model.train()

    #summary for current training loop and a running average object for loss
    loss_avg= utils.RunningAverage()

    train_data.cuda()
    train_data=train_data.contiguous()

    labels=train_label.contiguous()

    with tqdm(total=trsize) as pbar:
        for t in range(0, trsize, batch_size):

            inputs=torch.Tensor(batch_size,1,30,30,30).cuda()
            targets=torch.Tensor(batch_size,1,30,30,30).cuda()

            k=0
        
            for i in range(t, min(t+batch_size,trsize)):
	            input=train_data[i]
	            target=labels[i]
	            input.cuda()
	            target.cuda()

	            inputs[k] = input
	            targets[k]=target
	            k=k+1

            #clear previous gradients
            optimizer.zero_grad()

            #compute model output
            outputs=model.forward(inputs)

            #calculate loss
            loss = loss_fn(outputs,targets)

            #calculate gradients
            loss.backward()

            #perform updates using calculated gradients
            optimizer.step()

            #update the average loss
            loss_avg.update(loss.item())

            pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            pbar.update()
    logging.info("Average Loss on this epoch: {}".format(loss_avg()))


def train_epochs(model, optimizer, loss_fn, train_data, train_label, trsize, num_epochs, batch_size, model_dir,
                 restore_file=None, test_data=None, test_label=None):

    """
    Train the model for a certain number of epochs
    """

    #reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    for epoch in range(num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))

        #one full pass over the training set
        train(model, optimizer, loss_fn, train_data, train_label, trsize, batch_size)

        #evaluate for one epoch on validation set - do this or nah??
        if test_data != None:
            test_error = test.test_all(model, test_data, test_label, test_data.size(0))
            logging.info("Average Dice Similarity on test-set is {}".format(test_error))

        #save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              checkpoint=model_dir)


if __name__ == '__main__':

    args = parser.parse_args()

    #set the random seed for reproducible experiments
    torch.cuda.manual_seed(1234)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    #Load training data
    train_data, trsize= data_loader.load_data(args.train_data_dir, 'dataset')
    logging.info("Number of training examples: {}".format(trsize))

    #Load training labels
    train_label, lsize= data_loader.load_data(args.train_data_dir, 'labels')
    logging.info("Number of training labels: {}".format(trsize))

    #Load test data
    test_data, tesize= data_loader.load_data(args.test_data_dir, 'dataset')
    logging.info("Number of test examples: {}".format(tesize))

    #Load test labels
    test_label, telsize= data_loader.load_data(args.test_data_dir, 'labels')
    logging.info("Number of test labels: {}".format(telsize))

    #initialize autoencoder
    autoencoder= net.VolAutoEncoder()
    autoencoder.cuda()

    #define optimizer
    optimizer = optim.SGD(autoencoder.parameters(), lr=args.learning_rate, momentum=args.momentum)

    #fetch loss function
    loss_fn = net.loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.epochs))
    train_epochs(autoencoder, optimizer, loss_fn, train_data, train_label, trsize, args.epochs, args.batch_size, args.model_dir,
                       args.restore_file, test_data, test_label )
