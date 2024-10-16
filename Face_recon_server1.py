import argparse
import functools
import os
import time

import cv2
import numpy as np
import paddle
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments

import yaml
import uuid
import json
import requests
from datetime import timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug import run_simple
from gevent import pywsgi
from collections import OrderedDict
import shutil

app = Flask(__name__)
app.config['app.json.ensure_ascii'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)

        # 加载模型
        self.model = paddle.jit.load(mobilefacenet_model_path)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = {}
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            imgs = self.process(imgs)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            feature = self.infer(imgs[0])
            faces_db[name] = feature[0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, img):
        assert len(img.shape) == 3 or len(img.shape) == 4
        if len(img.shape) == 3:
            img = img[np.newaxis, :]
        img = paddle.to_tensor(img, dtype='float32')
        # 执行预测
        feature = self.model(img)
        return feature.numpy()

    def recognition(self, image_path, face_name):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        s = time.time()
        imgs, boxes = self.mtcnn.infer_image(img)
        print('人脸检测时间：%dms' % int((time.time() - s) * 1000))
        imgs = self.process(imgs)
        if imgs is None:
            return None, None
        imgs = np.array(imgs, dtype='float32')
        s = time.time()
        features = self.infer(imgs)
        print('人脸识别时间：%dms' % int((time.time() - s) * 1000))
        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i]
            results_dict = {}
            for name in self.faces_db.keys():
                feature1 = self.faces_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            print('人脸对比结果：', results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold and result[0] == result[0]:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names

    @staticmethod
    def add_text(img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simfang.ttf', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, image_path, boxes_c, names, rec_name):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] - 15, color=(0, 0, 255), size=12)
            filename = rec_name + '.jpg'
        cv2.im_show = Image.fromarray(img[:, :, ::-1], mode='RGB')
        cv2.im_show.save('./caches/' + filename)
        data = {'file': open('./caches/' + filename, 'rb')}
        image_data = requests.post('IP', files=data)
        image_data = image_data.json()
        print(image_data)
        Img_data = image_data['data']
        #Img_data = ""
        print(Img_data)
        return Img_data


def clean_caches():  # 清除上传的文件
    shutil.rmtree('./caches')
    os.mkdir('./caches')
    shutil.rmtree('./face_db')
    os.mkdir('./face_db')
    


@app.route('/face_ocr', methods=['POST', 'GET'])
def detect():
    try:
        files = request.files.getlist('files')
        uploaded_file_folder = './caches'
        name = request.form.get('name')
    except Exception as r:
        return jsonify({
            'msg': "上传失败",
            'code': 500,
            'data': {
                'Status': 'failed',
            }})
    try:
        for file in files:
            files = request.files.getlist('files')
            filename = file.filename
            save_path = os.path.join(uploaded_file_folder, filename)
            file.save(save_path)  # Save to caches, it should be clean after every session
            # ----------------------------DO NOT CHANGE----------------------------------
            parser = argparse.ArgumentParser(description=__doc__)
            add_arg = functools.partial(add_arguments, argparser=parser)
            add_arg('image_path', str, save_path, '预测图片路径')
            add_arg('face_db_path', str, 'face_db', '人脸库路径')
            add_arg('threshold', float, 0.6, '判断相识度的阈值')
            add_arg('mobilefacenet_model_path', str, 'models/infer/model', 'MobileFaceNet预测模型的路径')
            add_arg('mtcnn_model_path', str, 'models/mtcnn', 'MTCNN预测模型的路径')
            args = parser.parse_args()
            print_arguments(args)
            predictor = Predictor(mtcnn_model_path=args.mtcnn_model_path,
                                  mobilefacenet_model_path=args.mobilefacenet_model_path,
                                  face_db_path=args.face_db_path,
                                  threshold=args.threshold)
            start = time.time()
            boxes, names = predictor.recognition(args.image_path, name)
            count = 0
            for i in range(len(names)):
                if name not in names[i]:
                    count += 1
            if count == len(names):
                return jsonify({
                    'msg': "未识别到目标人脸",
                    'code': 200,
                    'data': {
                        'Status': 'success',
                        'Result': '未识别到目标人脸'
                    }})
            face_location = boxes.astype(np.int_).tolist()  # '预测的人脸位置：'
            face_name = names  # '识别的人脸名称：'
            Rec_time = (int((time.time() - start) * 1000))  # '总识别时间：%dms'
            for i in range(len(face_name)):
                img = predictor.draw_face(args.image_path, boxes, names, name)
            
        file_name = './caches/' + name + '.jpg'
        if file_name:
            clean_caches()
            return jsonify({
                'msg': '识别成功',
                'code': 200,
                'data': {
                    'Status': 'success',
                    'Face location': face_location,
                    'Rec_face_name': face_name,
                    'Time': str(Rec_time) + 'ms',
                    'Img': img,
                    'Result':'识别成功，已识别到目标人脸'
                },
            })
    except:
        return jsonify({
            'msg': "识别失败或无人脸",
            'code': 500,
            'data': {
                'Status': 'failed',
                'Result': '识别失败或无目标人脸' 
            }})


@app.route('/face_db', methods=['POST', 'GET'])
def facedb():  # upload face to face database
    try:
        file = request.files['file']
        uploaded_file_folder = './face_db'
        name = request.form.get('name')
        filename = name + '.jpg'
        print(filename)
        save_path = os.path.join(uploaded_file_folder, filename)
        file.save(save_path)  # save to face_db, this will be clean after every session
        print("人脸数据保存成功")
    except Exception as r:
        return jsonify({
            'msg': "上传失败",
            'code': 200,
            'data': {
                'Status': 'failed',
            }})
    return jsonify({
        'msg': "上传成功",
        'code': 200,
        'data': {
            'Status': 'success'
        }})


if __name__ == '__main__':
    #server = pywsgi.WSGIServer((IP, POTR), app)
    #server.serve_forever()
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, processes=1)
