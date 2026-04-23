import math
from pynput import keyboard
import threading
from mpc import SimpleMPC
import numpy as np
from mpc import MPC

class Controller:
    def __init__(self, dt=0.01):
        self.R = 0.4 
        self.r = 0.05

        self.steer_output = [0.0, 0.0, 0.0, 0.0]
        self.wheel_output = [0.0, 0.0, 0.0, 0.0]

        self.target = [0, 0, 0]  # vx vy vw

        self.mpc = MPC(dt)
        self.state = np.zeros(3)   # [X, Y, theta]
        self.ref = np.array([1.0, 1.0, 1.0])  # 目标点
        # self._start_keyboard()

    # 正运动学解算函数
    def forward_kinematics(self, vx, vy, w):
        self.steer_output[0] = math.atan2(vy + self.R*w, vx)
        self.wheel_output[0] = math.sqrt(vx**2 + (vy + self.R*w)**2) / self.r
        self.steer_output[1] = math.atan2(vy, vx - self.R*w) - math.pi / 2
        self.wheel_output[1] = math.sqrt((vx - self.R*w)**2 + vy**2) / self.r
        self.steer_output[2] = math.atan2(vy - self.R*w, vx) + math.pi
        self.wheel_output[2] = math.sqrt(vx**2 + (vy - self.R*w)**2) / self.r
        self.steer_output[3] = math.atan2(vy, vx + self.R*w) + math.pi / 2
        self.wheel_output[3] = math.sqrt((vx + self.R*w)**2 + vy **2) / self.r
        
        return self.steer_output, self.wheel_output
    
    def on_press(self, key):
        if key == keyboard.Key.up:
            self.target[1] = 100.0
        elif key == keyboard.Key.down:
            self.target[1] = -100.0
        elif key == keyboard.Key.left:
            self.target[0] = -100.0
        elif key == keyboard.Key.right:
            self.target[0] = 100.0
        elif key == keyboard.Key.shift_l:
            self.target[2] = -100.0
        elif key == keyboard.Key.shift_r:
            self.target[2] = 100.0

    def on_release(self, key):
        self.target[0] = 0
        self.target[1] = 0
        self.target[2] = 0

    # 键盘监听函数
    def _listen_keyboard(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
    
    def _start_keyboard(self):
        thread = threading.Thread(target=self._listen_keyboard)
        thread.start()

    # def update(self):
    #     self.forward_kinematics(*self.target)
    #     return self.steer_output, self.wheel_output
    
    def update_state(self, d):
        # 从 mujoco 读取 base 位姿（你需要根据模型改）
        self.state[0] = d.qpos[0]
        self.state[1] = d.qpos[1]
        self.state[2] = d.qpos[2]

    def update(self, d):
        self.update_state(d)

        u = self.mpc.solve(self.state, self.ref)

        vx, vy, w = u

        self.forward_kinematics(vx, vy, w)

        return self.steer_output, self.wheel_output