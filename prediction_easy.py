import carla
import time
import cv2
import numpy as np


# 连接到Carla模拟器
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town05')

# 生成自动驾驶车辆
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = carla.Transform(carla.Location(x=40, y=0, z=3), carla.Rotation(yaw=180))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)


try:
    while True:
        # 获取车辆当前位置和速度
        current_location = vehicle.get_location()
        current_velocity = vehicle.get_velocity()

                
        # 进行轨迹预测
        prediction_time = 2.0  # 预测时间
        predicted_location = current_location + current_velocity * prediction_time

        # 绘制预测轨迹
        color = carla.Color(r=0, g=0, b=255, a=150)
        world.debug.draw_line(current_location, predicted_location, thickness=1, color=color, life_time=0.2)

        time.sleep(0.1)

except KeyboardInterrupt:
    # 销毁传感器和车辆
    camera.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()