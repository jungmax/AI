"""
YOLOv8n 실시간 웹캠 객체 탐지

설치:
    pip install ultralytics opencv-python

사용법:
    python webcam.py                                   # 기본 사전학습 모델
    python webcam.py --weights runs/my_yolov8n/weights/best.pt   # 직접 학습한 모델
    python webcam.py --cam 1 --conf 0.5                # 다른 카메라, 신뢰도 0.5

종료: 'q' 키
스크린샷 저장: 's' 키
"""
import argparse
import time
import os
import cv2
from ultralytics import YOLO
import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt", help="모델 가중치 경로")
    p.add_argument("--cam", type=int, default=0, help="웹캠 번호 (기본 0)")
    p.add_argument("--conf", type=float, default=0.4, help="탐지 신뢰도 임계값")
    p.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    #카메라 실행 안될 때
    if not cap.isOpened():
        raise RuntimeError(f"웹캠 {args.cam}번을 열 수 없습니다.")
    
    prev_time = time.time()

    save_dir = "/Users/jungmax/sample/dataset"
    os.makedirs(save_dir, exist_ok=True)

    search_pattern = os.path.join(save_dir, "capture_*.jpg")
    existing_files = glob.glob(search_pattern)

    
    print(f"[{save_dir}] 폴더에서 찾은 기존 사진 개수: {len(existing_files)}개")
    shot_count = len(existing_files)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        # 추론 (verbose=False: 콘솔 출력 끔)
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # 박스/라벨이 그려진 프레임
        annotated = results[0].plot()

        # 탐지된 객체 정보 출력하고 싶을 때
        # for box in results[0].boxes:
        #     cls_id = int(box.cls[0])
        #     name = model.names[cls_id]
        #     conf = float(box.conf[0])
        #     x1, y1, x2, y2 = map(int, box.xyxy[0])
        #     print(name, conf, (x1, y1, x2, y2))

        # FPS 계산 및 표시
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        #카메라 텍스트 설정
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #노트북 모니터에 띄우는 코드
        cv2.imshow("YOLOv8n Webcam", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot_count += 1
            path = f"/Users/jungmax/sample/dataset/capture_{shot_count}.jpg"
            cv2.imwrite(path, annotated)
            print(f"저장됨: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
