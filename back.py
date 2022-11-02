"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import datetime
import io
import shutil
from logging.config import fileConfig
from pathlib import Path
import torch
from flask import send_file, Flask, request
from PIL import Image
from flask_cors import CORS, cross_origin
from flask_api import status
from flask_apscheduler import APScheduler
from os import walk

fileConfig('logging.cfg')

path_to_repo = Path().resolve()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
models = {'model1', 'model2', 'model3', 'model4'}
pathsDates = []

DETECTION_URL = "/api/image/run/<model>"

scheduler = APScheduler()
scheduler.init_app(app)


@scheduler.task('interval', id='deletePhotos', seconds=60, misfire_grace_time=300)
def deletePhotos():
    # print('Delete job executed')
    # print(pathsDates)
    for pd in pathsDates:
        try:
            if (datetime.datetime.now() - pd[1]).seconds > 10:
                shutil.rmtree(path_to_repo / pd[0])
                pathsDates.remove(pd)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (pd, e))
    # print(pathsDates)


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin()
def predict(model):
    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        if model in models:
            runnable = torch.hub.load(path_to_repo, model, path_to_repo / Path(model + '.pt'), source='local',
                                      skip_validation=True, device='cpu')
            results = runnable(im, size=640)
            path = results.save()
            del runnable
            del results
            del im
            del im_file
            del im_bytes
            pathsDates.append((path, datetime.datetime.now()))
            return send_file(path / next(walk(path), (None, None, []))[2][0], mimetype='image/jpeg')
    return "BAD REQUEST", status.HTTP_400_BAD_REQUEST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5005, type=int, help="port number")
    opt = parser.parse_args()
    scheduler.start()
    app.run(host="0.0.0.0", port=opt.port)
else:
    scheduler.start()
