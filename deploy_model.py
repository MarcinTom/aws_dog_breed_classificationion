import logging
import os
import sys
import subprocess

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Just to handle smdebug instalation as without it I'm getting missing module error
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def net(device):
    logger.info("Model creation for fine-tuning started.")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)

    model = model.to(device)
    logger.info("Model creation completed.")

    return model


def model_fn(model_dir):
    install('smdebug')
    device = "cpu"
    logger.info(f"Device: {device}")

    model = net(device)

    logger.info("Loading model weights")

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()

    return model