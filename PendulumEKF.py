import numpy as np


class PendulumEKF:
    """
    Drop-in replacement EKF (internally Error-State EKF)

    External interface is identical to your original class.
    """

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, I, a):

        self.I = I

        # adaptive model parameters
        self.a = a #0.943  #0.9
        self.fc = 0.0
        self.fv = 0.0
        self.v_eps = 0.1

        # ---- NOMINAL STATE (true estimate) ----
        self.x = np.zeros((2, 1))  # [theta, omega]

        # ---- ERROR covariance ----
        self.P = np.eye(2) * 0.1  #0.1

        # disturbance torque variance (MAIN tuning knob)
        self.q_torque = 100.0  #10.00

        # encoder noise
        self.R = np.array([[1e-4]])

        self.initialized = False

    # =====================================================
    @staticmethod
    def wrap_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    # =====================================================
    # NONLINEAR DYNAMICS (nominal propagation)
    # =====================================================
    def dynamics(self, x, u):

        theta = x[0, 0]
        omega = x[1, 0]

        friction = (
            self.fc * np.tanh(omega / self.v_eps)
            + self.fv * omega
        )

        theta_dd = (u - self.a*np.sin(theta) - friction) / self.I

        return np.array([
            [omega],
            [theta_dd]
        ])

    # =====================================================
    # ERROR DYNAMICS JACOBIAN
    # =====================================================
    def error_jacobian(self):

        theta = self.x[0, 0]
        omega = self.x[1, 0]

        sech2 = 1.0 - np.tanh(omega / self.v_eps) ** 2

        dfdtheta = -self.a*np.cos(theta) / self.I
        dfdomega = (
            -self.fc*sech2/self.v_eps - self.fv
        ) / self.I

        return np.array([
            [0.0, 1.0],
            [dfdtheta, dfdomega]
        ])

    # =====================================================
    def discretize(self, A, dt):
        Adt = A * dt
        return np.eye(2) + Adt + 0.5 * (Adt @ Adt)

    # =====================================================
    def process_noise(self, dt):
        # disturbance enters as unknown torque
        G = np.array([[0.0],
                      [1.0/self.I]])

        Qc = np.array([[self.q_torque]])

        return G @ Qc @ G.T * dt

    # =====================================================
    # MAIN UPDATE (UNCHANGED INTERFACE)
    # =====================================================
    def update(self, theta_meas, torque_u, dt):

        if dt <= 0:
            return self.x[0,0], self.x[1,0]

        # ---------- initialization ----------
        if not self.initialized:
            self.x[0,0] = theta_meas
            self.x[1,0] = 0.0
            self.initialized = True
            return self.x[0,0], self.x[1,0]

        # ====================================
        # 1️⃣ PROPAGATE NOMINAL STATE
        # ====================================
        xdot = self.dynamics(self.x, torque_u)
        self.x += dt * xdot
        self.x[0,0] = self.wrap_angle(self.x[0,0])

        # ====================================
        # 2️⃣ PROPAGATE ERROR COVARIANCE
        # ====================================
        A = self.error_jacobian()
        F = self.discretize(A, dt)
        Q = self.process_noise(dt)

        self.P = F @ self.P @ F.T + Q

        # ====================================
        # 3️⃣ MEASUREMENT UPDATE
        # ====================================
        H = np.array([[1.0, 0.0]])

        innovation = self.wrap_angle(theta_meas - self.x[0,0])
        y = np.array([[innovation]])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # error-state correction
        dx = K @ y

        # inject correction into nominal state
        self.x += dx
        self.x[0,0] = self.wrap_angle(self.x[0,0])

        # Joseph stabilized covariance
        I2 = np.eye(2)
        self.P = (
            (I2 - K @ H) @ self.P @ (I2 - K @ H).T
            + K @ self.R @ K.T
        )

        return self.x[0,0], self.x[1,0]

    # =====================================================
    # ADAPTIVE PARAMETER UPDATE (SAFE VERSION)
    # =====================================================
    def update_parameters(self, a=None, fc=None, fv=None):

        changed = False

        if a is not None:
            self.a = float(a)
            changed = True

        if fc is not None:
            self.fc = float(fc)
            changed = True

        if fv is not None:
            self.fv = float(fv)
            changed = True

        # 🔥 CRITICAL:
        # adaptive laws change system dynamics → inflate uncertainty
        if changed:
            self.P += np.diag([5e-3, 2e-2])
