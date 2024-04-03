from flask import Flask, request, render_template, Response
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load your trained TensorFlow Lite model
tflite_interpreter = tf.lite.Interpreter(model_path='../models/model.tflite')
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

@app.route('/', methods=['GET'])
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from post request
    file = request.files['file']

    # Read the image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    # Set the model input
    tflite_interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))

    # Run inference
    tflite_interpreter.invoke()

    # Get the prediction
    prediction = tflite_interpreter.get_tensor(output_details[0]['index'])

    # Process your result for human
    pred_processed = (prediction > 0.5).astype(np.uint8)

    # Convert to binary image and return it
    _, encoded_img = cv2.imencode('.PNG', np.squeeze(pred_processed) * 255)
    return Response(encoded_img.tobytes(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
