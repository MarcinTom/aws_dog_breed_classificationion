import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

import argparse
import logging
import sys
import os

from PIL import ImageFile
import smdebug.pytorch as smd
from smdebug.pytorch import get_hook
from smdebug.pytorch import modes
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

hook = get_hook(create_if_not_exists=True)
logger.info(f"Hook {hook}")

def test(model, test_loader, criterion, device, logger, hook):
    '''
    Test the model
    '''
    if hook:
        hook.set_mode(modes.EVAL)
        
    model.eval()  # Set the model to evaluation mode
    correct = 0
    test_loss = 0
    with torch.no_grad():  # No gradients are needed for the evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the device
            output = model(data)  # Forward pass
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    # Calculate the percentage of correct answers
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Loss: {:.4f}\n".format(test_loss)
        )
    test_accuracy = 100 * correct / len(test_loader.dataset)
    logger.info(f'Test set: Accuracy: {test_accuracy:.2f}%')


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, logger , hook):
    logger.info("Training started.")
    for i in tqdm(range(epochs), desc="Training"):

        train_loss = 0
        train_correct = 0
        model.train()

        for data, target in train_loader:
            if hook:
                    hook.set_mode(modes.TRAIN)
                
            data = data.to(device) # Move data and target to the device
            target = target.to(device) # Move data and target to the device

            optimizer.zero_grad()

            outputs = model(data) # Runs Forward Pass

            loss = criterion(outputs, target) # Calculates Loss

            loss.backward() # Calculates Gradients for Model Parameters
            optimizer.step() # Updates Weights

            train_loss += loss.item() # Sum up batch loss

            # Calculate prediction & accuracy
            pred = outputs.max(1, keepdim=True)[1] # Get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()  # Counts number of correct predictions

        logger.info(f"Epoch {i}: Train size = {len(train_loader.dataset)}")
        train_loss /= len(train_loader.dataset) # Average loss per batch
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset) # Average accuracy
        logger.info(f"Epoch {i}: Train loss = {train_loss:.4f}")
        logger.info(f"Epoch {i}: Train accuracy = {train_accuracy}%")

        val_loss = 0
        val_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in validation_loader:
                if hook:
                    hook.set_mode(modes.EVAL)
                    
                data = data.to(device) # Move data and target to the device
                target = target.to(device) # Move data and target to the device

                outputs = model(data) # Runs Forward Pass
                # Calculate & sum batch loss
                loss = criterion(outputs, target) # Calculates Loss
                val_loss += loss.item() # Sum up batch loss

                # Calculate prediction & accuracy
                pred = outputs.max(1, keepdim=True)[1] # Get the index of the max log-probability
                val_correct += pred.eq(target.view_as(pred)).sum().item()  # Counts number of correct predictions

            val_loss /= len(validation_loader.dataset) # Average loss per batch
            val_accuracy = 100.0 * val_correct / len(validation_loader.dataset) # Average accuracy
            logger.info(f"Epoch {i}: Valid size = {len(validation_loader.dataset)}")
            logger.info(f"Epoch {i}: Val loss = {val_loss:.4f}")
            logger.info(f"Epoch {i}: Validation accuracy = {val_accuracy}%")


    logger.info("Training completed.")


def net():
    '''
    Initialises pretrained model
    '''
    logger.info("Model initialisation started")

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features # default number of neurons in resnet50 is 2048
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(num_features, 133) # 133 is the number of dog classes

    return model



def create_data_loaders(data, batch_size, test_batch_size):
    '''
    Dataloader function
    '''
    train_data_path = os.path.join(data, 'train') # Calling OS Environment variable and split it into 3 sets
    test_data_path = os.path.join(data, 'test')
    valid_data_path = os.path.join(data, 'valid')
    
    # Define the training data transformations
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flips the image horizontally with a probability of 0.5
        transforms.Resize((224,224)),
        transforms.ToTensor(),                   # Converts the image to a PyTorch tensor with values between 0 and 1
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor image with mean and std for each channel
    ])

    # Define the testing data transformations
    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),                   # Converts the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor image similarly to the training data
    ])

    # Choose your dataset (for example MNIST), ensure to replace with your dataset of choice
    trainset = ImageFolder(train_data_path, transform=training_transform)
    valset = ImageFolder(valid_data_path, transform=testing_transform)
    testset = ImageFolder(test_data_path, transform=testing_transform)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader  = torch.utils.data.DataLoader(valset, test_batch_size, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def main(args):
    '''
    Initialize a model by calling the net function
    '''

    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model.to(device) # Ensure the model is on the correct device
    logger.info("Data loader creation started")
    train_loader, val_loader, test_loader = create_data_loaders(args.data_dir,  args.batch_size, args.test_batch_size)
    logger.info("Data loader creation completed")

    # Create a hook if not exist
    # hook = None
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    if hook:
        hook.register_loss(loss_criterion)
        
    # Call the train function to start training your model
    logger.info("Model training started")
    train(model, train_loader, val_loader, loss_criterion, optimizer, args.epochs, device, logger, hook)
    logger.info("Model training completed")

    # Test the model to see its accuracy
    logger.info("Model testing started")
    test(model, test_loader, loss_criterion, device, logger, hook)
    logger.info("Model testing completed")

  
    # Save the trained model
    logger.info("Saving the model")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, path)
    logger.info("Model data saved")

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # Specify all the hyperparameters you need to use to train your model.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for testing (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, metavar="LR", help="learning rate (default: 1e-2)"
    )

    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    args=parser.parse_args()

    main(args)
