import torch

from mnist_classfifier import MnistClassfier
from model import Mnist


model = Mnist()
model.load_state_dict(torch.load("save_model/model.pt"))

mnist_classifier_service = MnistClassfier()
mnist_classifier_service.pack('model', model)
saved_path = mnist_classifier_service.save()