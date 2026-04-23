import time
import mujoco
import mujoco.viewer

from controller import Controller
from write_sensor import RobotWriter
from mpc import SimpleMPC
from mpc import MPC

# 加载模型
m = mujoco.MjModel.from_xml_path("scence.xml")
d = mujoco.MjData(m)

controller = Controller(dt=m.opt.timestep)
writer = RobotWriter(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 控制器计算
        steer_out, wheel_out = controller.update(d)

        # 写入 MuJoCo
        writer.write(steer_out, wheel_out)

        mujoco.mj_step(m, d)
        viewer.sync()

        time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))