"""
학습용 사진 촬영 (박스 없는 원본 저장)

사용법:
    python collect.py
    python collect.py --cam 1 --out dataset/raw

조작:
    스페이스바 : 사진 1장 저장
    a 키       : 자동 촬영 on/off (1초마다 1장)
    q 키       : 종료
"""
import argparse
import os
import time

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--out", default="dataset/raw", help="저장 폴더")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise RuntimeError(f"웹캠 {args.cam}번을 열 수 없습니다.")

    count = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])
    auto, last_shot = False, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 저장은 원본(frame)으로, 화면 표시용 복사본에만 글씨를 넣음
        view = frame.copy()
        status = f"saved: {count} | auto: {'ON' if auto else 'OFF'}"
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        # 실제 화면에 띄워주는 명령어
        cv2.imshow("Collect (SPACE=shot, a=auto, q=quit)", view)

        key = cv2.waitKey(1) & 0xFF
        shoot = False
        if key == ord(" "):
            shoot = True
        elif key == ord("a"):
            auto = not auto
        elif key == ord("q"):
            break

        if auto and time.time() - last_shot > 1.0:
            shoot = True

        if shoot:
            count += 1
            last_shot = time.time()
            cv2.imwrite(os.path.join(args.out, f"img_{count:04d}.jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"총 {count}장 저장됨 -> {args.out}")


if __name__ == "__main__":
    main()
