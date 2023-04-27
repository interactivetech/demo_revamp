
# Demo Revamp Steps
Allows you to swap Roboflow datasets easily, and swap Detectron2-based models easily

* Roboflow datasets supported: `x-ray-rheumatology`, `flir-camera-objects`
* Models: FasterRCNN Resnet50 FPN 1x, MaskRCNN Resnet50 FPN 1x, Cascade RCNN Resnet50 FPN 3x

# Steps to Run

## Spin up JupyterLab notebook
Use the default launch jupyter notebook settings.
## Install Dependencies
* `apt-get update && apt-get install nano`
## Install Dependencies
* `git clone https://github.com/interactivetech/demo_revamp.git`
* `cd demo_revamp/determined/examples/computer_vision/detectron2_coco_pytorch/`
* `bash startup-hook.sh`

# Create or Login to roboflow

Follow these instructions so get your API KEY: https://docs.roboflow.com/rest-api#obtaining-your-api-key

# Open notebook: `Demo_Revamp.ipynb`
In * `demo_revamp/determined/examples/computer_vision/detectron2_coco_pytorch/`
In cell run the cell with python code, where you
Update   <API_KEY> with your private API key and run cell: `!python download_datasets.py --dataset-dir /run/determined/workdir/shared_fs/data/ --key <API_KEY>`

You can now run the rest of the cells that has the `det e` commands to train MaskRCNN and FasterRCNN on the `x-ray-rheumatology` and `flir-camera-objects`


# Optional: Run training on `x-ray-rheumatology`
det  experiment create -f const.yaml .
# Optional: Run training on `flir-camera-objects`
Flir: experiment create -f const-flir.yaml .


