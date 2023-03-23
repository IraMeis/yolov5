"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import datetime
import gc
import glob
import io
import os
import shutil
from logging.config import fileConfig
from pathlib import Path
import torch
from flask import send_file, Flask, request, jsonify
from PIL import Image
from flask_cors import CORS, cross_origin
from flask_api import status
from flask_apscheduler import APScheduler

fileConfig('logging.cfg')

path_to_repo = Path().resolve()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
models = {'model1', 'model2', 'model3', 'model4'}
pathsDates = []

DETECTION_URL = "/api/image/run/<model>"
EVAL_URL = "/api/image/eval/<name>"

scheduler = APScheduler()
scheduler.init_app(app)


@scheduler.task('interval', id='deletePhotos', seconds=60, misfire_grace_time=300)
def deletePhotos():
    # print('Delete job executed')
    # print(pathsDates)
    for pd in pathsDates:
        try:
            if (datetime.datetime.now() - pd[1]).seconds > 20:
                shutil.rmtree(path_to_repo / pd[0])
                pathsDates.remove(pd)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (pd, e))
    # print(pathsDates)


@app.route(DETECTION_URL, methods=["POST"])
@cross_origin(expose_headers=['Content-Disposition'])
def predict(model):
    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        if model in models:
            runnable = torch.hub.load(path_to_repo, model, path_to_repo / Path(model + '.pt'), source='local',
                                      skip_validation=True)
            results = runnable(im, size=640)
            path = results.save()
            results._run(txt=True, save_dir=path_to_repo / path)
            del runnable
            del results
            del im
            del im_file
            del im_bytes
            gc.collect()
            torch.cuda.empty_cache()
            pathsDates.append((path, datetime.datetime.now()))
            old_file = glob.glob(str(path) + '/*.jpg')[0]
            new_file = os.path.join(path_to_repo / path,  os.path.basename(os.path.normpath(path_to_repo / path)) + '.jpg')
            os.rename(old_file, new_file)
            return send_file(new_file, mimetype='image/jpeg', as_attachment=True)
    return "BAD REQUEST", status.HTTP_400_BAD_REQUEST


@app.route(EVAL_URL, methods=["GET"])
@cross_origin()
def eval_model(name):
    mainPath = path_to_repo / 'runs' / 'detect' / name
    dam_count = {'D00': 0, 'D10': 0, 'D20': 0, 'D40': 0, 'ALL': 0}
    try:
        with open(glob.glob(str(mainPath) + "/*.txt")[0]) as f:
            for line in f:
                (val, key) = line.split()
                dam_count[key] = int(val)
            for k, v in dam_count.items():
                dam_count['ALL'] += dam_count[k] * int(k != 'ALL')
        return jsonify(dam_count)

    except IndexError:
        return jsonify(dam_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5005, type=int, help="port number")
    opt = parser.parse_args()
    scheduler.start()
    app.run(host="0.0.0.0", port=opt.port)
else:
    scheduler.start()
