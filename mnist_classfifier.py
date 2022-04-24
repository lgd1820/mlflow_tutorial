import torch

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.pytorch import PytorchModelArtifact

from model import Mnist

@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('model')])
class MnistClassfier(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, input):
        inputs = torch.Tensor(input[0]["image"])
        return self.artifacts.model.predict(inputs)