import torch

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.pytorch import PytorchModelArtifact

from model import Mnist

@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('model')])
class MnistClassfier(BentoService):
    @api(input=JsonInput())
    def predict(self, inputs):
        inputs = torch.Tensor(inputs["image"])
        return self.artifacts.model(inputs)