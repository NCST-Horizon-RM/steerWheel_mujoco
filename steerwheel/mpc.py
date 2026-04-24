# MPC.py
import numpy as np

class SimpleMPC:
    def __init__(self, dt):
        self.dt = dt

        # 权重（可以调）
        self.Q = np.diag([10.0, 10.0, 5.0])   # 状态误差
        self.R = np.diag([0.1, 0.1, 0.1])     # 控制输入

    def linearize(self, theta):
        A = np.eye(3) + np.array([
            [0, 0, -self.dt * np.sin(theta) - self.dt * np.cos(theta)],
            [0, 0,  self.dt * np.cos(theta) - self.dt * np.sin(theta)],
            [0, 0, 0]
        ])

        B = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ]) * self.dt

        return A, B

    def solve(self, x, x_ref):
        """
        x: 当前状态 [X, Y, theta]
        x_ref: 参考状态
        """

        theta = x[2]
        A, B = self.linearize(theta)

        # 离散 LQR（近似 MPC）
        P = self.Q.copy()

        for _ in range(10):  # Riccati迭代
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            P = self.Q + A.T @ P @ (A - B @ K)

        # 控制律
        u = -K @ (x - x_ref)

        return u  # [vx, vy, w]
    


class MPC:
    def __init__(self, dt, N=10):
        self.dt = dt
        self.N = N  # prediction horizon

        self.Q = np.diag([10.0, 10.0, 10.0])
        self.R = np.diag([0.1, 0.1, 0.1])

    # =========================
    # 线性化（你这个版本 ✔）
    # =========================
    def linearize(self, theta):
        A = np.eye(3) + np.array([
            [0, 0, -self.dt * np.sin(theta)],
            [0, 0,  self.dt * np.cos(theta)],
            [0, 0, 0]
        ])

        B = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ]) * self.dt

        return A, B

    # =========================
    # 构建预测矩阵
    # =========================
    def build_prediction(self, A, B):
        n = A.shape[0]
        m = B.shape[1]

        Phi = np.zeros((n*self.N, n))
        Gamma = np.zeros((n*self.N, m*self.N))

        A_power = np.eye(n)

        for i in range(self.N):
            Phi[i*n:(i+1)*n] = A_power
            A_power = A @ A_power

        for i in range(self.N):
            for j in range(i+1):
                A_power = np.linalg.matrix_power(A, i-j)
                Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = A_power @ B

        return Phi, Gamma

    # =========================
    # 求解 MPC（无约束）
    # =========================
    def solve(self, x, x_ref):
        theta = x[2]
        A, B = self.linearize(theta)

        Phi, Gamma = self.build_prediction(A, B)

        Q_bar = np.kron(np.eye(self.N), self.Q)
        R_bar = np.kron(np.eye(self.N), self.R)

        x_ref_bar = np.tile(x_ref, self.N)

        # cost: ||Phi x + Gamma u - x_ref||^2 + u^T R u
        H = Gamma.T @ Q_bar @ Gamma + R_bar
        f = Gamma.T @ Q_bar @ (Phi @ x - x_ref_bar)

        # solve: H u = -f
        U = -np.linalg.solve(H, f)

        u0 = U[:3]  # 只取第一个控制量

        return u0