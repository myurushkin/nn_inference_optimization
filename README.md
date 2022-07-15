# Measurements of pytorch inference performance

Experements that help to check the real performance of neural network inference and compare it with theoretical performance for particular gpu.

Parameters:
 * gpu: gtx1070
 * nn: simple_network | yolov5x (default) | resnet50

Report generated for default settings:

```
NN GFLOPs: 77.1287745
GPU peak performance (GFLOPS): 6500.0
Average NN execution time (sec): 0.04161948627895779
Theoretical minimal inference time (sec): 0.011865965307692308
Inference relative performance 28.51%
```
