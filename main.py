import time
import torch
from torch import nn
from ptflops import get_model_complexity_info
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 10000),
            nn.Linear(10000, 10000),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def measure(model, input):
    start = time.time()
    model(input)
    end = time.time()
    return end - start


if __name__ == "__main__":
    with torch.cuda.device(0):
        net = NeuralNetwork()
        macs, params = get_model_complexity_info(net, (100,), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        input = np.zeros((1, 100), dtype=np.float32)
        input = torch.tensor(input)

        results = [measure(net, input) for _ in range(10)]
        everage_time = np.mean(results)

    gpuPeakPerformance = 8.141 * 1e12
    theoretical_nn_inference_time = macs / gpuPeakPerformance

    print("NN GFLOPs: {0}".format(macs / 1e9))
    print("GPU peak performance (GFLOPS): {0}".format(gpuPeakPerformance / 1e9))
    print("Average NN execution time (sec): {0}".format(everage_time))
    print("Theoretical minimal inference time (sec): {0}".format(theoretical_nn_inference_time))
