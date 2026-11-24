"""
YOLOv8n 커스텀 데이터셋 학습 코드

설치:
    pip install ultralytics

사용법:
    python train.py
"""
from ultralytics import YOLO
import torch


def main():
    # GPU 있으면 0번 GPU, 없으면 CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    # 사전학습된 yolov8n 가중치 로드 (없으면 자동 다운로드)
    model = YOLO("yolov8n.pt")

    # 학습 시작
    model.train(
        data="data.yaml",   # 데이터셋 설정 파일
        epochs=100,         # 학습 횟수
        imgsz=640,          # 입력 이미지 크기
        batch=16,           # 배치 크기 (GPU 메모리 부족하면 8, 4로 줄이기)
        device=device,
        workers=4,          # Windows에서 오류 나면 0으로
        patience=20,        # 20 epoch 동안 개선 없으면 조기 종료
        project="runs",     # 결과 저장 폴더
        name="my_yolov8n",  # 실험 이름
        pretrained=True,
        # 데이터 증강 (기본값도 충분하지만 필요시 조절)
        # hsv_h=0.015, fliplr=0.5, mosaic=1.0,
    )

    # 검증 데이터로 성능 평가
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # (선택) 다른 포맷으로 내보내기: onnx, engine(TensorRT) 등
    # model.export(format="onnx")

    print("학습 완료! 가중치 위치: runs/my_yolov8n/weights/best.pt")


if __name__ == "__main__":  # Windows에서 멀티프로세싱 때문에 필수
    main()
