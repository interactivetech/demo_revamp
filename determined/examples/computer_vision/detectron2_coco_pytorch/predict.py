from detectron2.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
import torch
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

roboflow_dataset_dirs = {
        'x-ray-rheumatology': '/run/determined/workdir/shared_fs/data/x-ray-rheumatology',
        'flir-camera-objects': '/run/determined/workdir/shared_fs/data/flir-camera-objects',
    }

def register_roboflow_dataset(dataset_name,cfg):
    '''
    '''
    cfg.defrost()
    DIR = roboflow_dataset_dirs[dataset_name]
    print("DIR: ",DIR)
    TRAIN_JSON_PATH = os.path.join(DIR,'train/_annotations.coco.json')
    TRAIN_IMG_DIR = os.path.join(DIR,'train/')
    VAL_JSON_PATH = os.path.join(DIR,'valid/_annotations.coco.json')
    VAL_IMG_DIR = os.path.join(DIR,'valid/')
    TEST_JSON_PATH = os.path.join(DIR,'test/_annotations.coco.json')
    TEST_IMG_DIR = os.path.join(DIR,'test/')
    train_dataset_name = "train_{}".format(dataset_name)
    val_dataset_name="val_{}".format(dataset_name)
    test_dataset_name = "test_{}".format(dataset_name)
    register_coco_instances(train_dataset_name, {}, TRAIN_JSON_PATH, TRAIN_IMG_DIR)
    register_coco_instances(val_dataset_name, {}, VAL_JSON_PATH, VAL_IMG_DIR)
    register_coco_instances(test_dataset_name, {}, TEST_JSON_PATH, TEST_IMG_DIR)
    cfg.DATASETS.TRAIN=("train_{}".format(dataset_name),)
    cfg.DATASETS.VAL=("val_{}".format(dataset_name),)
    cfg.DATASETS.TEST=("test_{}".format(dataset_name),)
    # cfg.freeze()
    return cfg, train_dataset_name, val_dataset_name, test_dataset_name


def predict(ckpt_path,
            img_path,
            confidence=0.05,
            yaml_path=None,
           dataset_name=None):

    ckpt = torch.load(os.path.join(ckpt_path, "state_dict.pth"))['models_state_dict'][0]

    torch.save(ckpt,os.path.join(ckpt_path,'stripped_ckpt.pth'))# have to strip the models_state_dict and list to get state dict to load in Detectron
    dataset_name = dataset_name
    cfg = get_cfg()
    DatasetCatalog.clear()
    cfg, train_dataset_name, val_dataset_name, test_dataset_name = register_roboflow_dataset(dataset_name,cfg)
    YAML_PATH = yaml_path

    cfg.merge_from_file(YAML_PATH)
    
    nuts_metadata = MetadataCatalog.get(train_dataset_name)
    dataset_dicts = DatasetCatalog.get(train_dataset_name)
    cfg.MODEL.WEIGHTS = os.path.join(ckpt_path, "stripped_ckpt.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    predictor = DefaultPredictor(cfg)
    inputs = cv2.imread(img_path)
    outputs = predictor(inputs)
    print(outputs)
    v = Visualizer(inputs[:, :, ::-1],
                   metadata=nuts_metadata,
                   scale=1.2,
                   instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    return outputs