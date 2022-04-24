import torchvision.datasets as dsets
import torchvision.transforms as transforms
import requests

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

inputs = mnist_test.data[0].view(-1, 28*28).tolist()

datas = {"image":inputs}

req = requests.post("http://127.0.0.1:5000/predict", json=datas)
print(req.json())