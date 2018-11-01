import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
from torchvision import models
import torch.nn.functional as F
from torch import nn, optim

from PIL import Image
import glob, re, os, copy, random, time

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')