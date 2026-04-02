## 舵轮简单仿真

![image](./robot/screenshot.png)

> 后续会把麦轮和全向也加入。

### 文件
```bash
-- robot            # 舵轮带小云台
    -- controller.py  # 控制器
    -- scence.xml     # 仿真文件
    -- ...
-- steerwheel       # 舵轮底盘
    -- controller.py  # 控制器
    -- scence.xml     # 仿真文件
```

### 问题
- 小陀螺原地旋转，舵向电机1会一段时间转2pi角度导致偏心严重。
  
是vy解算中一会是0.0,一会是-0.0导致的。加上下面的抽象代码即解决。

```python
if vy == -0.0:
    vy = 0.0
```

### 启动
``` bash
cd robot
python3 controller.py
```