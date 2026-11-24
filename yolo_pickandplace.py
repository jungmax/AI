import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

# MoveIt2 파이썬 인터페이스 (moveit_py 또는 헬퍼 라이브러리 가정)
# 실제 환경의 패키지 구조에 맞추어 임포트를 조정하여 사용하세요.
from moveit_py import MoveItPy, PlanRequestParameters

class YoloPickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('yolo_pick_and_place_node')
        
        # 1. 학습된 커스텀 YOLO 모델 로드
        self.model = YOLO('best.pt')
        self.bridge = CvBridge()
        
        # 2. 웹캠 또는 로봇 카메라 스트림 오픈 (또는 ROS Image 토픽 서브스크라이브)
        self.cap = cv2.VideoCapture(0)
        
        # 3. MoveIt2 초기화 (로봇 팔 그룹 이름: 'manipulator' 또는 'co_bot' 등)
        # self.robot = MoveItPy(node_name="moveit_py_picker")
        # self.arm_group = self.robot.get_planning_component("manipulator")
        
        self.get_logger().info("YOLOv8 + MoveIt2 Pick & Place 노드가 준비되었습니다.")
        
        # 0.1초 주기로 카메라 프레임 분석 및 제어 루프 실행
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # YOLOv8 추론 실행 (신뢰도 50% 이상만 필터링)
        results = self.model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # 감지된 객체(Boxes) 순회
        for box in results[0].boxes:
            # 1. 바운딩 박스 중심점(Pixel) 계산
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            u = int((xyxy[0] + xyxy[2]) / 2)
            v = int((xyxy[1] + xyxy[3]) / 2)
            
            # 클래스 이름 확인 (예: airpod, laptop 등)
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            
            # 2. [핵심] 픽셀 좌표를 로봇 기준 3D 좌표(X, Y, Z)로 변환
            # (주의: 실제 구현 시 Depth 카메라나 실린더/평면 방정식을 이용해 3D 변환을 거쳐야 합니다)
            target_pose_3d = self.pixel_to_robot_coordinates(u, v)
            
            if target_pose_3d:
                self.get_logger().info(f"감지된 물체: {class_name} | 목표 위치: {target_pose_3d}")
                
                # 3. MoveIt2를 통한 Pick 동작 수행 (일회성 실행을 위해 주석 해제 후 연동)
                # self.execute_pick_motion(target_pose_3d)
                # rclpy.shutdown() # 테스트용으로 한 개 집으면 종료

        # 실시간 화면 시각화
        cv2.imshow("YOLOv8 + MoveIt2 Vision", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def pixel_to_robot_coordinates(self, u, v):
        """
        카메라 픽셀 좌표(u, v)를 로봇 베이스 좌표계(X, Y, Z)로 변환하는 함수.
        Depth 카메라 정보를 활용하거나 고정 캘리브레이션 행렬을 곱해 계산합니다.
        """
        # 예시용 더미 변환 로직 (실제 뎁스 센서 연동 필요)
        target_x = 0.3 + (u - 320) * 0.001
        target_y = 0.0 + (v - 240) * 0.001
        target_z = 0.05  # 물체 높이
        return [target_x, target_y, target_z]

    def execute_pick_motion(self, pose):
        """
        MoveIt2 API를 이용해 해당 3D 위치로 로봇 팔을 이동시키고 그리퍼를 닫는 동작
        """
        self.get_logger().info(f"MoveIt2 경로 계획 및 픽업 시작: {pose}")
        # 여기에 MoveIt2 목표 포즈 설정, 경로 계획(Plan), 실행(Execute) 코드가 들어갑니다.
        pass

def main(args=None):
    rclpy.init(args=args)
    node = YoloPickAndPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()