from roboflow import Roboflow
import os
rf = Roboflow(api_key=os.environ["ROBOFLOW_KEY"])
project = rf.workspace("roboflow-100").project("flir-camera-objects")
dataset = project.version(2).download("coco",location="/run/determined/workdir/shared_fs/data/flir-camera-objects")