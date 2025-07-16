import os
import firebase_config
import uuid
from firebase_admin import storage, firestore
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from dotenv import load_dotenv
import pytz

report_id = uuid.uuid4().hex
user_id = "admin"

# 업로드할 파일 경로
local_file_path = "output/kickboard_id3_frame164_conf0.72.jpg"

# 파일 확장자 추출
_, file_extension = os.path.splitext(local_file_path)

# Storage에 저장할 경로 생성
storage_path = f"Report/{report_id}{file_extension}"

# Storage에 파일 업로드
bucket = storage.bucket()
blob = bucket.blob(storage_path)
blob.upload_from_filename(local_file_path)
blob.make_public()
file_url = blob.public_url

# 위도경도 추출
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        return None
    exif = {}
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            exif[decoded] = gps_data
        else:
            exif[decoded] = value
    return exif

def get_lat_lon(exif_data):
    def _convert_to_degrees(value):
        # IFDRational 객체를 float로 변환
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + m/60 + s/3600

    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None, None
    
    lat = _convert_to_degrees(gps_info["GPSLatitude"])
    if gps_info["GPSLatitudeRef"] != "N":
        lat = -lat
        
    lon = _convert_to_degrees(gps_info["GPSLongitude"])
    if gps_info["GPSLongitudeRef"] != "E":
        lon = -lon
        
    return lat, lon

exif = get_exif_data(local_file_path)
# lat, lon = get_lat_lon(exif)
lat, lon = "37.3005","127.0392"

# 한국 시간대 객체 생성
kst = pytz.timezone('Asia/Seoul')

# Firestore 클라이언트 가져오기
db_fs = firestore.client()

data = {
    "date": datetime.now(kst),
    "gpsInfo": f"{lat} {lon}",
    "imageUrl": file_url,
    "userId": "CCTV Report",
    "violation": ["CCTV 실시간 감지"]
}

# Firestore에 저장 (컬렉션: Report, 문서: report_id)
db_fs.collection("Report").document(report_id).set(data)
