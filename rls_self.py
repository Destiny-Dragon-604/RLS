import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
from PendulumEKF import PendulumEKF

# NOTE:
# If your helper file is named SVM_helper_functions2.py, change this import accordingly:
# from SVM_helper_functions2 import MinSVD
from SVM_helper_functions2 import MinSVD

import csv


################## CSV LOGGER ##################
class CsvLogger:
    def __init__(
        self,
        log_dir="data",
        prefix="pendulum",
        fieldnames=None,
        flush_every=50,
        flush_interval_s=0.5,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.csv"

        self.fieldnames = fieldnames or [
            "t_s",          # time since calibration (s)
            "p_turns",      # pendulum angle (turns)
            "p_rad",        # pendulum angle (rad)
            "v_turns_s",    # pendulum velocity (your v is rad/s; keeping name)
            "phi_rad_s",    # wheel speed (rad/s)
            "a_est",        # your estimator state
            "u_torque",     # commanded torque
            "raw_turns",    # raw encoder turns [0..~1)
            "rest_turns",   # calibration rest position
            "hturns",       # wrap counter
            "dt_s",         # loop dt
            "x_d_s",        # desired state store
            "rho_s",        # auxiliary state
            "SV_s",         # singular value
            "accel_s",      # accel (I * p_ddot)
            "eq_trim_deg",  # AUTO TRIM (deg)
            "phi_filt_rad_s",  # filtered wheel speed used for trim
        ]

        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._w.writeheader()

        self._buf = []
        self._flush_every = int(flush_every)
        self._flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()

    def log(self, **row):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._buf.append(clean)

        now = time.monotonic()
        if len(self._buf) >= self._flush_every or (now - self._last_flush) >= self._flush_interval_s:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        self._w.writerows(self._buf)
        self._buf.clear()
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self):
        try:
            self.flush()
        finally:
            self._f.close()


################## ENCODER ##################
bus = smbus.SMBus(1)

def read_raw_angle():
    data = bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255 + data[1] / 64 / 255


################# WAIIIITTT #################
async def wait_for_enter(prompt: str = "Press ENTER to start..."):
    await asyncio.to_thread(input, prompt)


################# Filters ####################
class DirtyDerivative:
    """
    Low-pass filtered numerical derivative:
        dx = a*dx + (1-a)*(x - x_prev)/dt
    where a = tau/(tau + dt), tau = 1/(2*pi*fc)
    """
    def __init__(self, fc_hz: float, x0: float = 0.0):
        self.fc_hz = float(fc_hz)
        self.tau = 1.0 / (2.0 * np.pi * self.fc_hz)
        self.x_prev = float(x0)
        self.dx = 0.0
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.dx

        if not self.initialized:
            self.x_prev = float(x)
            self.dx = 0.0
            self.initialized = True
            return self.dx

        a = self.tau / (self.tau + dt)
        dx_raw = (x - self.x_prev) / dt
        self.dx = a * self.dx + (1.0 - a) * dx_raw
        self.x_prev = float(x)  # IMPORTANT
        return self.dx


class LowPass1:
    """1st-order low-pass: y <- y + alpha*(x - y), alpha = dt/(tau+dt), tau=1/(2πfc)."""
    def __init__(self, fc_hz: float, y0: float = 0.0):
        self.fc_hz = float(fc_hz)
        self.tau = 1.0 / (2.0 * np.pi * self.fc_hz)
        self.y = float(y0)
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.y
        if not self.initialized:
            self.y = float(x)
            self.initialized = True
            return self.y
        a = dt / (self.tau + dt)
        self.y = self.y + a * (float(x) - self.y)
        return self.y


################## Regressor ##################
def phi_regressor(theta, angular_velocity):
    v_esp = 0.1
    return np.array([np.sin(theta), -0*np.tanh(angular_velocity / v_esp), -0*angular_velocity])


################## ODrive helpers ##################
async def get_odrive_velocity_rad_s(odrive) -> float:
    """
    Returns motor velocity in rad/s.
    Tries async method first, then common attributes.
    """
    if hasattr(odrive, "get_velocity"):
        v = await odrive.get_velocity()      # usually turns/s
    elif hasattr(odrive, "velocity"):
        v = odrive.velocity                  # usually turns/s
    elif hasattr(odrive, "vel_estimate"):
        v = odrive.vel_estimate              # usually turns/s
    else:
        v = 0.0
    return float(v) * 2.0 * np.pi            # turns/s -> rad/s


################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0.002)

    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)
    next_time = 0
    dt_target = 0.001

    SVD = MinSVD(p_max=5, eta=0.25) #5 .05

    # Gains
    C = 6  #6
    k_rho = 1
    k_a = 1*0.01
    gamma = 10
    pi = np.pi
    I = 0.037322 
    k_s = 0.5 # 0
    a_actual = 1.006
    ekf = PendulumEKF(I,a_actual)
    u=0


    # -------- AUTO TRIM (smoothed + gated + slew-limited) --------
    vel_lp = LowPass1(fc_hz=0.5)   # SLOWER = less jumpy (captures drift, not ripple)  was 0.5

    trim_switch = 1
    eq_trim_rad = 0.0
    eq_trim_rad1 = 0.0
    eq_trim_deg = 0.0
    kp_trim = 0.01
    trim_k_i = 0.0001         # integrator gain (start small)
    trim_deadband = 0.001     # rad/s: ignore small velocity noise
    trim_max_deg = 1.00      # absolute trim clamp
    trim_slew_deg_s = 0.005  # max trim change rate (deg/s) -> kills jumps

    # ---- Calibration ----
    rest_pos = read_raw_angle()

    print("\nSystem initialized. Move pendulum near upright (doesn't need to be perfect).")
    await wait_for_enter("Press ENTER to start control (zero reference will be set now)... ")

    t0 = time.monotonic()

    # ---- CSV log rate control ----
    log_frequency = 30
    log_period_s = 1 / log_frequency
    last_log_s = 0.0

    logger = CsvLogger(prefix="pendulum_control")
    print(f"[CSV] Logging to: {logger.path}")

    # Initialize unwrap state
    val = read_raw_angle()
    hturns = 0

    # (kept from your script)
    odrive.set_torque(0)

    # Adaptive state init
    a_last = 0.0#0.943
    fc_last = 0.0
    fv_last = 0.0
    xi_last = np.array([a_last, fc_last, fv_last]).reshape(-1, 1)

    loop = asyncio.get_running_loop()
    t_prev = loop.time()
    
# ---- Desired trajectory ----
    freq = .8
    omega = freq * 2 * pi
    A_deg = 5
            #if t_s > 20.0:  #5.0
            #    A_deg = 10
            #elif t_s > 15.0:  #5.0
            #    A_deg = 5   
            #elif t_s > 10.0:
            #    A_deg = 10
    A = (A_deg / 180.0) * pi


    try:
        while datetime.now() < stop_at:
            next_time += dt_target
            await asyncio.sleep(max(0, next_time - loop.time()))
            # ---- Encoder unwrap ----
            prev_val = val
            val = read_raw_angle()
            diff = val - prev_val

            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns
            p_meas = (position - rest_pos) * pi     # rad (your original scaling)
            p = p_meas - eq_trim_rad1                # APPLY AUTO TRIM

            # ---- Timing ----
            t_now = loop.time()
            dt = t_now - t_prev
            if dt <= 0:
                dt = 1e-6

            # ---- Desired trajectory ----
            #freq = .8
            #omega = freq * 2 * pi
            #A_deg = 30
            #if t_s > 20.0:  #5.0
            #    A_deg = 10
            #elif t_s > 15.0:  #5.0
            #    A_deg = 5   
            #elif t_s > 10.0:
            #    A_deg = 10
            #A = (A_deg / 180.0) * pi
            t_s = t_now - t0
            
            
            SIN_Wt = np.sin(-omega * t_s)
            offset = (0.1) * pi / 180.0 * 0
            x_d = A * SIN_Wt + offset
            xd_d = -A * omega * np.cos(-omega * t_s)
            xdd_d = -A * omega * omega * SIN_Wt

            # ---- State estimates ----
            e = p - x_d
            p_hat, v = ekf.update(p, u, dt)

            theta_DD = ekf.dynamics(
                np.array([[p_hat],[v]]),
                u
            )[1,0]

            ed = v - xd_d
            rho = ed + C * e

            # ---- ODrive velocity + AUTO TRIM update (smooth + gated + slew limited) ----
            phi = await get_odrive_velocity_rad_s(odrive)  # rad/s
            phi_filt = vel_lp.update(phi, dt)

            # Only adjust trim when system is relatively calm
            steady = (abs(v) < 0.5) and (abs(rho) < 0.5)

            if steady and abs(phi_filt) > trim_deadband:
                # integrator suggestion
                dtrim = -trim_k_i * (phi_filt ) * dt

                # slew limit
                dtrim_max = (trim_slew_deg_s * pi / 180.0) * dt
                if dtrim > dtrim_max:
                    dtrim = dtrim_max
                elif dtrim < -dtrim_max:
                    dtrim = -dtrim_max
                
                eq_trim_rad += dtrim 
                eq_trim_rad1 =trim_switch*( eq_trim_rad - kp_trim*phi_filt)

                # clamp absolute trim
                trim_max_rad = trim_max_deg * pi / 180.0
                if eq_trim_rad1 > trim_max_rad:
                    eq_trim_rad1 = trim_max_rad
                elif eq_trim_rad1 < -trim_max_rad:
                    eq_trim_rad1 = -trim_max_rad

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # ---- Adaptive update ----
            Phi = phi_regressor(x_d, v) #normally used p
            SUM = SVD.sum_components()
            if SUM.shape[0] == 0 and SUM.shape[1] == 0:
                SUM = np.array([0, 0, 0]).reshape(-1, 1)

            xi = xi_last + dt * (gamma * Phi.reshape(-1, 1) * rho - 1 * gamma * k_a * SUM)

            # ---- Control law ----
            u_ad = Phi @ xi
            u = ( -xdd_d *I
                + k_rho * rho
                + k_s * np.sign(rho)
                + 1 * e
                + I * C * ed
                + 1 * u_ad[0]
                - 0.00 * phi
            )

            odrive.set_torque(u)
            accel = I * theta_DD

            # ---- Log (rate-limited) ----
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s,
                    p_turns=p / (2 * pi),
                    p_rad=p,
                    v_turns_s=v,
                    phi_rad_s=phi,
                    a_est=xi[0][0],
                    u_torque=u,
                    raw_turns=val,
                    rest_turns=rest_pos,
                    hturns=hturns,
                    dt_s=dt,
                    x_d_s=x_d,
                    rho_s=rho,
                    SV_s=SVD.SV,
                    accel_s=accel,
                    eq_trim_deg=eq_trim_deg,
                    phi_filt_rad_s=phi_filt,
                )
                last_log_s = t_s

            # ---- Update estimator memory ----
            if t_s < 10.0:
                _SV = SVD.update(phi=Phi[0], epsilon=-(-I * theta_DD + u - u_ad[0]))
                
            xi_last = xi
            t_prev = t_now
            
            #ekf.update_parameters(
            #    a=xi[0][0],
            #    fc=xi[1][0],
            #    fv=xi[2][0]
            #)
            

            print(
                f"a est = {xi[0][0]:.3f},"
                f"fc est = {xi[1][0]:.3f},"
                f"fv est = {xi[2][0]:.3f},"
                f"p (deg) = {p*180/pi:.2f},"
                f"SV = {SVD.SV:.2f},"
                f"trim (deg) = {eq_trim_deg:+.2f},"
                f"phi_filt (rad/s) = {phi_filt:+.2f},"
            )

            await asyncio.sleep(0)

    finally:
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        logger.close()
        print("[CSV] Logger closed.")


################## ODRIVE SETUP ##################
odrive = pyodrivecan.ODriveCAN(0)


async def main():
    odrive.clear_errors(identify=False)
    print("Cleared Errors")
    await asyncio.sleep(1)

    odrive.initCanBus()

    print("Put Arm at bottom center to calibrate Zero Position.")
    await asyncio.sleep(1)
    cur_pos = odrive.position
    await asyncio.sleep(1)
    print(f"Encoder Absolute Position Set: {cur_pos}")

    odrive.setAxisState("open_loop_control")

    await asyncio.gather(
        odrive.loop(),
        controller(odrive),
    )


################ RUN #################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping...")
        try:
            odrive.set_torque(0)
        except Exception:
            pass
        try:
            odrive.estop()
        except Exception:
            pass
