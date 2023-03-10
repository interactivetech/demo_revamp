
# Demo Revamp Steps
Allows you to swap Roboflow datasets easily, and swap Detectron2-based models easily

* Roboflow datasets supported: `x-ray-rheumatology`, `flir-camera-objects`
* Models: FasterRCNN Resnet50 FPN 1x, MaskRCNN Resnet50 FPN 1x, Cascade RCNN Resnet50 FPN 3x

# Steps to Run
`apt-get update && apt-get install nano`
`git clone https://github.com/facebookresearch/detectron2 detectron2_repo`
`pip install -e detectron2_repo`
`pip install opencv-python`
`apt-get update && apt-get install ffmpeg libsm6 libxext6  -y`
`export ROBOFLOW_KEY=<INSERT KEY>`
`pip install roboflow`
`python download_roboflow_dataset.py`
`pip install setuptools==59.5.0`

`export DET_MASTER=<URL>`

# Run training on `x-ray-rheumatology`
det  experiment create -f const.yaml .
# Run training on `flir-camera-objects`
Flir: experiment create -f const-flir.yaml .


# Detectron2 Example

This example is a port of Detectron2 using Determined's PyTorchTrial API. The original example can be found on
 [Facebook Research's Detectron2 Github](https://github.com/facebookresearch/detectron2/blob/v0.6/tools/plain_train_net.py). More information on the original benchmarks can be found at the benchmark [documentation] (https://detectron2.readthedocs.io/en/latest/notes/benchmarks.html) page.

## Files
* **model_def.py**: The core code for the model. This includes building and compiling the model.
* **detectron2_files/**: This folder includes original Detectron2 files that have been slightly altered to work with Determined.


### Configuration Files
* **const.yaml**: Train the model with constant hyperparameter values.
* **distributed.yaml**: Same as `const.yaml`, but trains the model with multiple GPUs (distributed training).
* **const_fake.yaml**: This is used in Determined's automated test workflows and can be ignored.

## Data
The Common Objects in Context (COCO) dataset is the primary dataset for this port. Information on how to download the dataset can be found [here](https://cocodataset.org/#home). Once downloaded to the host machine, configure the bind mount in the experiment configuration files to point to the location. More information on bind mounts can be found [here](https://docs.determined.ai/latest/tutorials/data-access.html#distributed-file-system).

## To Run
If you have not yet installed Determined, installation instructions can be found
under `docs/install-admin.html` or at https://docs.determined.ai/latest/index.html

Run the following command: `det -m <master host:port> experiment create -f 
const.yaml .`. The other configurations can be run by specifying the appropriate 
configuration file in place of `const.yaml`.

## Results
The default configurations are based on [the original Detectron benchmarks](https://github.com/facebookresearch/detectron2/blob/v0.6/configs/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x.yaml). At the end of training, the boxAP and and segmAP should be over 34.00.

## Environment
We provide a [Docker image](https://hub.docker.com/r/determinedai/example-detectron2) with CUDA 10.1, PyTorch 1.10, other Determined dependencies, and Detectron2 0.6 - it is partially based on the [Detectron2 Dockerfile](https://github.com/facebookresearch/detectron2/blob/v0.6/docker/Dockerfile), but notably with OpenCV omitted.

