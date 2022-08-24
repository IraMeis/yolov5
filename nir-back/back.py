
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

path_to_repo = 'D:\\nirProjectBase\\yolo\\yolov5\\'
path_to_models = path_to_repo + 'models\\'
app = Flask(__name__)
models = {}
modelNames = {'model1','model2','model3','model4'}

DETECTION_URL = "/api/nets/run/<model>"


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    if request.method != "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)
            return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', nargs='+', default=[
        'model1',
        'model2',
        'model3',
        'model4'
    ], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()
    print(opt.model)
    for m in opt.model:
        models[m] = torch.hub.load(path_to_repo, m, path_to_models + m + '.pt', source='local', force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)
                                                   