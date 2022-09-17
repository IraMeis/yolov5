
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
from flask_api import status
from os import walk

path_to_repo = Path().resolve()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
models = {}
paths = []

DETECTION_URL = "/api/nets/run/<model>"


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin()
def predict(model):
    
    for p in paths:
        try:
            shutil.rmtree(path_to_repo / p)
            paths.remove(p)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (p, e))

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)
            path = results.save()
            paths.append(path)
            return send_file(path / next(walk(path), (None, None, []))[2][0], mimetype='image/jpeg')

    return "BAD REQUEST", status.HTTP_400_BAD_REQUEST


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
        models[m] = torch.hub.load(path_to_repo, m, path_to_repo / Path(m + '.pt'), source='local', force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)