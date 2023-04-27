import argparse
from roboflow import Roboflow
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download roboflow datasets')
    parser.add_argument('--dataset-dir', default='/run/determined/workdir/shared_fs/data/', help='path to checkpoint')
    parser.add_argument('--key', default='/run/determined/workdir/shared_fs/data/', help='path to checkpoint')
    args = parser.parse_args()
    rf = Roboflow(api_key=args.key)
    os.makedirs(os.path.dirname(args.dataset_dir), exist_ok=True)
    project = rf.workspace("roboflow-100").project("x-ray-rheumatology")
    dataset = project.version(2).download("coco",location=os.path.join(args.dataset_dir,"x-ray-rheumatology"))
    print(f"Creating dataset: ",os.path.join(args.dataset_dir,"x-ray-rheumatology"))
    project = rf.workspace("roboflow-100").project("flir-camera-objects")
    dataset = project.version(2).download("coco",location=os.path.join(args.dataset_dir,"flir-camera-objects"))
    print(f"Creating dataset: ",os.path.join(args.dataset_dir,"flir-camera-objects"))