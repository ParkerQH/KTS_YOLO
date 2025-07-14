import os
import cv2
import numpy as np
import requests
import tempfile
from ultralytics import YOLO
from firebase_admin import storage, firestore

# YOLO 모델 로드
model_kickboard = YOLO('YOLO/kickboard_yolov11s.pt')
model_person    = YOLO('YOLO/person_yolov11m.pt')
model_helmet    = YOLO('YOLO/helmet_yolov11m.pt')
model_brand     = YOLO('YOLO/kickboardBrand_yolov11m.pt')