
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import shutil
from pathlib import Path

import torch
from flask import send_file, Flask, request
from PIL import Image
from flask_cors import CORS, cross_origin
from os import walk

path_to_repo = 'D:\\nirProjectBase\\yolo\\yolov5\\'
path_to_models = path_to_repo + 'models\\'
path_to_result = path_to_repo + 'nir-back\\runs\\detect\\exp\\'
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
models = {}

DETECTION_URL = "/api/nets/run/<model>"


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin()
def predict(model):
    if request.method != "POST":
        return
    try:
        shutil.rmtree(path_to_result)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path_to_result, e))

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)
            results.save(save_dir=path_to_result)
            filenames = next(walk(path_to_result), (None, None, []))[2]
            return send_file(Path(path_to_result) / filenames[0], mimetype='image/jpeg')


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
    for m in opt.model:
        models[m] = torch.hub.load(path_to_repo, m, path_to_models + m + '.pt', source='local', force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)
