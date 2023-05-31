
# Demo Revamp (Computer Vision) Steps
This repo consists of training object detection models in determined, where you can swap datasets and models. The Object Detection models are based on Detectron2, and Datasets in this tutorial are using Roboflow.


* Roboflow datasets supported: `x-ray-rheumatology`, `flir-camera-objects`
* Models: FasterRCNN Resnet50 FPN 1x, MaskRCNN Resnet50 FPN 1x, Cascade RCNN Resnet50 FPN 3x

# Prereq
Tested using ds-workshops.determined.ai, T4 GPUs

# Steps to Run
* Create notebook environments with determined image: `determinedai/example-detectron2:0.6-cuda-10.2-pytorch-1.10`
* git clone repo
* cd demo_revamp/determined/examples/computer_vision/detectron2_coco_pytorch/configs

## Install Dependencies
* `apt-get update && apt-get install nano`
* `bash install-dep.sh`

# Create Roboflow account, get API key

Follow these instructions so get your API KEY: https://docs.roboflow.com/rest-api#obtaining-your-api-key



# Download datasets in shared_fs
Update   <API_KEY> with your private API key and run the below python script in a terminal:
* `!python download_datasets.py --dataset-dir /run/determined/workdir/shared_fs/data/ --key <API_KEY>`


# Open notebook: `Demo_Revamp.ipynb` to see and run premade yamls, swapping datasets and models
Here you can run through all the cells and see how to easily Train the following models:
* FasterRCNN on a FLIR Object Detection Dataset
* FasterRCNN on X Ray Rheumatology dataset
* MaskRCNN on a FLIR Object Detection Dataset
* MaskRCNN on X Ray Rheumatology dataset

## Example of Dataset swapping and Model Swapping
Run the `Demo_Revamp.ipynb` to see how to swap datasets and models. 

## Manually Swap Datasets

This repo can only toggle between two datasets, `x-ray-rheumatology` and `flir-camera-objects`. You can manually change the dataset by creating a new `const.yaml` editing the `dataset_name` yaml entry  to only have `x-ray-rheumatology` or `flir-camera-objects`

```yaml
name: detectron2_const_fasterrcnn_flir
environment:
    image: "determinedai/example-detectron2:0.6-cuda-10.2-pytorch-1.10"
hyperparameters:
  global_batch_size: 8 # Detectron defaults to 16 regardless of N GPUs
  model_yaml: models/fast_rcnn_R_50_FPN_1x.yaml
  dataset_name: 'flir-camera-objects'
  output_dir: None
  fake_data: False
searcher:
  name: single
  metric: bboxAP
  max_length: 
    batches: 9000
  smaller_is_better: false
resources:
    slots_per_trial: 4
entrypoint: model_def:DetectronTrial
max_restarts: 0
min_validation_period:
  batches: 100
```

## Manually Swap Models

This repo can only toggle between models defined in the `models/` directory. Currently there are 3 models supported: 
* FasterRCNN which is defined in `fast_rcnn_R_50_FPN_1x.yaml`
* MaskRCNN which is defined in `mask_rcnn_R_50_FPN_noaug_1x.yaml`
* CascadeRCNN, which is defined in `cascade_mask_rcnn_R_50_FPN_3x.yaml`

You can manually change the dataset by creating a new `const.yaml` editing the `model_yaml` yaml entry to only have `fast_rcnn_R_50_FPN_1x.yaml`,`mask_rcnn_R_50_FPN_noaug_1x.yaml`, `cascade_mask_rcnn_R_50_FPN_3x.yaml`
