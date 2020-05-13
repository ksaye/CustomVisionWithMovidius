"""
    CustomVision.AI ONNX file OpenVINO (for Movidius Acceleration)
"""

import os
import io
import sys
import onnxruntime
import datetime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
import tempfile
import json

# Imports for the REST API
from flask import Flask, request, jsonify

# Imports for image procesing
from PIL import Image
MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'

app = Flask(__name__)

# 4MB Max image size limit
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

@app.route('/')
def index():
    return 'CustomVision.ai model host harness for ONNX'

# Like the CustomVision.ai Prediction service /image route handle images
#     - octet-stream image file 
@app.route('/image', methods=['POST'])
@app.route('/<project>/image', methods=['POST'])
@app.route('/<project>/image/nostore', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/image', methods=['POST'])
@app.route('/<project>/classify/iterations/<publishedName>/image/nostore', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/image', methods=['POST'])
@app.route('/<project>/detect/iterations/<publishedName>/image/nostore', methods=['POST'])
def predict_image_handler(project=None, publishedName=None):
    try:
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        elif ('imageData' in request.form):
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())

        image = Image.open(imageData)
        predictions = od_model.predict_image(image)
        return str('{"created": "' + str(datetime.datetime.utcnow().isoformat()) + '", "id": "", "iteration": "' + manifest["IterationId"] + '", "predictions": ' + str(predictions) + ', "project": ""}').replace("'", "\"")
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            onnxruntime.set_default_logger_severity(0)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

with open(LABELS_FILENAME, 'r') as f:
    labels = [l.strip() for l in f.readlines()]

manifest = json.loads(open("cvexport.manifest").read())

od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

app.run(host='0.0.0.0', port=87)