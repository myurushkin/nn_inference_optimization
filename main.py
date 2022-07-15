import time, hydra, torch
from torch import nn
from ptflops import get_model_complexity_info
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torchvision.models as models


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 10000),
            nn.Linear(10000, 10000),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def measure(model, input):
    start = time.time()
    model(input)
    end = time.time()
    return end - start


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print("----------------------------")
    print("nn.name: {0}".format(cfg.nn.name))
    print("gpu.TFLOPS: {0}".format(cfg.gpu.TFLOPS))

    batch_size = 32
    with torch.cuda.device(0):
        if cfg.nn.name == 'simple_network':
            net = NeuralNetwork()
            input_tensor_shape = (1, 100)
            input = np.zeros(input_tensor_shape, dtype=np.float32)
        if cfg.nn.name == 'yolov5x':
            net = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, verbose=False)
            input_tensor_shape = (3, 640, 480)
            input = np.expand_dims(np.zeros(input_tensor_shape, dtype=np.float32), 0)
        if cfg.nn.name == 'resnet50':
            net = models.resnet50()
            input_tensor_shape = (3, 224, 224)
            input = np.expand_dims(np.zeros(input_tensor_shape, dtype=np.float32), 0)

        net = net.cuda()
        input = torch.tensor(input).cuda()
        results = [measure(net, input) for _ in range(cfg.experiments_count)]
        everage_time = np.mean(results[1:])

        macs, params = get_model_complexity_info(net, input_tensor_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)

    gpuPeakPerformance = cfg.gpu.TFLOPS * 1e12
    theoretical_nn_inference_time = macs / gpuPeakPerformance

    print("NN GFLOPs: {0}".format(macs / 1e9))
    print("GPU peak performance (GFLOPS): {0}".format(gpuPeakPerformance / 1e9))
    print("Average NN execution time (sec): {0}".format(everage_time))
    print("Theoretical minimal inference time (sec): {0}".format(theoretical_nn_inference_time))
    print("Inference relative performance {0:.2f}%".format(theoretical_nn_inference_time / everage_time * 100))


if __name__ == "__main__":
    my_app()
