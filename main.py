import torch

from mnist_classifier import MnistClassifier
from model import Mnist


model = Mnist()
model.load_state_dict(torch.load("save_model/model.pt"))

mnist_classifier_service = MnistClassifier()
mnist_classifier_service.pack('model', model)
saved_path = mnist_classifier_service.save()

