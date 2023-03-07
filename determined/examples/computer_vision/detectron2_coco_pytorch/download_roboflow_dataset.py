from roboflow import Roboflow
import os
rf = Roboflow(api_key=os.environ["ROBOFLOW_KEY"])
project = rf.workspace("roboflow-100").project("x-ray-rheumatology")
dataset = project.version(2).download("coco")