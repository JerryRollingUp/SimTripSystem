# pip install Flask tensorflow numpy Pillow PIL

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 加载TensorFlow模型
model = tf.saved_model.load('./tf_lung_class')

@app.route('/', methods=['GET'])
def index():
    # 返回一个上传表单
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Cannot find the file!'
    file = request.files['file']
    if file.filename == '':
        return 'Please choose a lung cancer image! :)'
    if file:
        file = file.read()
        image = tf.image.decode_jpeg(file, channels=3)
        image = tf.image.resize(image, [256, 256])
        # image = image / 255.0  # 归一化到[0, 1]
        image = tf.expand_dims(image, 0)  # 增加批次维度

        # 使用模型进行预测
        prediction = model(image)
        label_dict = {0:'Colon adenocarcinoma',1:'Colon benign tissue',2:'Lung adenocarcinoma',3:'Lung benign tissue',4:'Lung squamous cell carcinoma'}
        predicted_class = np.argmax(prediction, axis=1)
        results = label_dict[predicted_class[0]]

        # 返回预测结果
        return f'The lung cancer type is: {results}'

if __name__ == '__main__':
    app.run(debug=True)
