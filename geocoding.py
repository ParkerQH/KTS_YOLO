import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 위도/경도 값을 받아와 지번주소 반환
def gps(gps_info):
    lat_str, lon_str = gps_info.strip().split()
    lat = float(lat_str)
    lon = float(lon_str)
    parcel_addr = reverse_geocode(lat, lon,  os.getenv("VWorld_API"))
    return parcel_addr

# V-WORLD API 사용
def reverse_geocode(lat, lon, api_key):
    url = "https://api.vworld.kr/req/address"
    params = {
        "service": "address",
        "request": "getAddress",
        "crs": "epsg:4326",
        "point": f"{lon},{lat}",
        "format": "json",
        "type": "parcel",
        "key": api_key,
    }
    response = requests.get(url, params=params)

    # 반환값 단순화
    if response.status_code == 200:
        data = response.json()
        if data["response"]["status"] == "OK":
            # 첫 번째 결과에서 지번주소 추출
            result = data["response"]["result"][0]
            if "text" in result:
                return result["text"]  # 지번주소만 반환
    return None