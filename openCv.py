import cv2
import os
# 동영상 프레임별로..

# 저장 폴더 생성 함수
def make_output_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 영상에서 10프레임마다 이미지 추출 함수
def extract_frames(video_path, output_dir, frame_interval=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('영상 파일을 열 수 없습니다.')
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'총 프레임 수: {total_frames}')
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f'frame_{count:05d}.jpg')
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()
    print(f'저장된 이미지 수: {saved}')

if __name__ == '__main__':
    video_path = 'DJI_0086.mp4'  # 입력 영상 파일명
    output_dir = 'output_frames2'  # 저장 폴더명
    make_output_dir(output_dir)
    extract_frames(video_path, output_dir, frame_interval=10)
