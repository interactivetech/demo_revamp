git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
pip install opencv-python
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
export ROBOFLOW_KEY=1y1x15UFcVopfieUHsKX
pip install roboflow
python download_roboflow_dataset.py
pip install setuptools==59.5.0