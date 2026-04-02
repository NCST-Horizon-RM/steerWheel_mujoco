import time
import math
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
import threading
from quat import YawTracker
from pid import PID_control

dir_path = "scence.xml"
m = mujoco.MjModel.from_xml_path(dir_path)
d = mujoco.MjData(m)

ctrl = np.zeros(m.nu)

# 舵轮和轮子的关节名称
steer_joints = [f"steer{i}" for i in range(4)]
wheel_joints = [f"wheel{i}" for i in range(4)]

# 获取关节id
steer_ids = [m.joint(name).id for name in steer_joints]
wheel_ids = [m.joint(name).id for name in wheel_joints]

orientation_sensor_id = m.sensor('orientation').id
imu_chassis_id = m.sensor('imu_chassis').id

imu_chassis = YawTracker()
imu_yaw = YawTracker()

yaw_target = 0
yaw_pid = PID_control(1, 0, 1, yaw_target)

# 机器人参数（轮距、轴距、轮半径等）
R = 0.4  # 轴距
r = 0.05 # 轮半径
steer_output = [0.0, 0.0, 0.0, 0.0]  # 舵轮
wheel_output = [0.0, 0.0, 0.0, 0.0]  # 轮子

target = [0, 0, 0]

# 正运动学解算函数
def forward_kinematics(vx, vy, w):
    steer_output[0] = math.atan2(vy + R*w, vx)
    wheel_output[0] = math.sqrt(vx**2 + (vy + R*w)**2) / r
    steer_output[1] = math.atan2(vy, vx - R*w) - math.pi / 2
    wheel_output[1] = math.sqrt((vx - R*w)**2 + vy**2) / r
    steer_output[2] = math.atan2(vy - R*w, vx) + math.pi
    wheel_output[2] = math.sqrt(vx**2 + (vy - R*w)**2) / r
    steer_output[3] = math.atan2(vy, vx + R*w) + math.pi / 2
    wheel_output[3] = math.sqrt((vx + R*w)**2 + vy **2) / r
    
    return steer_output, wheel_output

def back_kinematics(steer_angles, wheel_speeds):
    pass

def on_press(key):
    if key == keyboard.Key.up:
        target[1] = 5.0
    elif key == keyboard.Key.down:
        target[1] = -5.0
    elif key == keyboard.Key.left:
        target[0] = -5.0
    elif key == keyboard.Key.right:
        target[0] = 5.0
    elif key == keyboard.Key.shift_l:
        target[2] = -10.0
    elif key == keyboard.Key.shift_r:
        target[2] = 10.0
    elif key == keyboard.Key.alt_l:
        target_yaw -= 0.1
    elif key == keyboard.Key.alt_r:
        target_yaw += 0.1
    # print(1)
    
def on_release(key):
    target[0] = 0
    target[1] = 0
    target[2] = 0

# 键盘监听函数
def listen_keyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# 启动键盘监听的线程
keyboard_thread = threading.Thread(target=listen_keyboard)
keyboard_thread.daemon = True  # 确保主程序退出时监听器也会退出
keyboard_thread.start()

with mujoco.viewer.launch_passive(m, d) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        # 读取当前舵轮角度和轮子速度
        steer_angles = [d.qpos[steer_ids[i]] for i in range(4)]
        wheel_speeds = [d.qvel[wheel_ids[i]] for i in range(4)]
        # 正运动学解算
        # forward_kinematics(target[0], target[1], target[2])
    
        adr_gimbal = m.sensor_adr[orientation_sensor_id]
        quat_gimbal = d.sensordata[adr_gimbal:adr_gimbal + 4]
        yaw_gimbal = imu_yaw.get_euler(quat_gimbal)
        
        adr_chassis = m.sensor_adr[imu_chassis_id]
        quat_chassis = d.sensordata[adr_chassis:adr_chassis + 4]
        yaw_chassis = imu_chassis.get_euler(quat_chassis)
        # print(yaw_chassis)
        
        err_angle = yaw_gimbal - yaw_chassis
        vx = target[0] * math.cos(err_angle) + target[1] * math.sin(err_angle)
        vy = -target[0] * math.sin(err_angle) + target[1] * math.cos(err_angle)
        
        forward_kinematics(vx, vy, 10.0)
        
        d.ctrl[8] = -yaw_pid.position_pid(yaw_target, yaw_gimbal)
        
        # 控制器：将目标写入控制量
        for i in range(4):
            d.ctrl[i] = steer_output[i]  # 前4个是舵轮位置控制
            d.ctrl[4 + i] = wheel_output[i]  # 后4个是轮子速度控制
            
            
        mujoco.mj_step(m, d)
        viewer.sync()
        # 控制仿真步长
        time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))