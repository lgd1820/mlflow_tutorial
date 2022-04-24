from bentoml import artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import JsonInput

@artifacts([PytorchModelArtifact('model')])
class MnistClassifier(BentoService):
    @api(input=JsonInput()):
    def predict(self, parsed_jsons):
        