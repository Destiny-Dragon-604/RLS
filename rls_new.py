"""
Pendulum Controller — Sliding Mode + 1-param RLS (estimates a = mgl)

Root cause of previous a_hat freeze:
    d_rls = u - I * theta_DD
    But theta_DD came from EKF dynamics which already contained A_ACTUAL*sin(p)/I.
    So d_rls = u - I*(u/I - A_ACTUAL*sin(p)/I) = A_ACTUAL*sin(p)  <- constant!
    RLS saw zero innovation because gravity was already cancelled by the EKF.

Fix:
    EKF uses A_ACTUAL=0 (pure integrator) for state estimation only.
    RLS residual is computed from a RAW finite-difference acceleration,
    with NO model gravity term subtracted:
        v_dot_raw = (v - v_prev) / dt
        d_rls     = u - I * v_dot_raw     <- contains full a*sin(p) signal
    Now RLS has a real signal to fit.
"""

import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
from PendulumEKF import PendulumEKF
import csv


# ─── Tuning ───────────────────────────────────────────────────────────────────
DT_TARGET      = 0.001
I_PENDULUM     = 0.037322

# EKF uses a=0 intentionally — pure integrator smoother, no gravity model
# so that gravity residual is fully visible to RLS
EKF_A          = 0.5

TRAJ_FREQ_HZ   = 0.8
TRAJ_AMP_DEG   = 10.0

C_SLIDE        = 6.0
K_RHO          = 1.0
K_S            = 0
K_E            = 1.0

# RLS
RLS_LAMBDA     = 0.999        # slightly slower forgetting for stability
RLS_P0         = 100.0
A_HAT_INIT     = 0.3         # initial guess, inside physical bounds
A_HAT_MIN      = 0.2
A_HAT_MAX      = 3.0

# v_dot low-pass to reduce noise in RLS residual
VDOT_LP_FC_HZ  = 20.0         # Hz — filter raw accel before feeding RLS

# Auto-trim
TRIM_SWITCH     = 1
TRIM_K_I        = 0.0001
TRIM_KP         = 0.01
TRIM_DEADBAND   = 0.001
TRIM_MAX_DEG    = 1.00
TRIM_SLEW_DEG_S = 0.005
TRIM_LP_FC_HZ   = 0.5

MAX_TORQUE     = 2.0
RUN_DURATION_S = 10000
LOG_FREQ_HZ    = 30


# ─── RLS (1 parameter) ────────────────────────────────────────────────────────
class RLS1:
    def __init__(self, lambda_f=0.998, p0=1000.0, theta_init=1.0):
        self.lambda_f = float(lambda_f)
        self.theta    = float(theta_init)
        self.P        = float(p0)

    def update(self, phi: float, d: float):
        denom      = self.lambda_f + phi * self.P * phi
        K          = (self.P * phi) / denom
        error      = d - self.theta * phi
        self.theta = self.theta + K * error   # never overwrite with clamped value
        self.P     = (self.P - K * phi * self.P) / self.lambda_f
        return self.theta, error


# ─── Filters ──────────────────────────────────────────────────────────────────
class LowPass1:
    def __init__(self, fc_hz: float, y0: float = 0.0):
        self.tau = 1.0 / (2.0 * np.pi * float(fc_hz))
        self.y   = float(y0)
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.y
        if not self.initialized:
            self.y = float(x)
            self.initialized = True
            return self.y
        a = dt / (self.tau + dt)
        self.y += a * (float(x) - self.y)
        return self.y


# ─── CSV logger ───────────────────────────────────────────────────────────────
class CsvLogger:
    FIELDS = [
        "t_s", "p_rad", "v_rad_s",
        "x_d", "xd_d", "e", "ed", "rho",
        "a_hat", "a_hat_raw", "rls_error",
        "u_ad", "u_torque",
        "phi_rad_s", "phi_filt_rad_s",
        "eq_trim_deg", "vdot_raw", "d_rls",
        "dt_s",
    ]

    def __init__(self, log_dir="data", prefix="pendulum"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.csv"
        self._f   = open(self.path, "w", newline="")
        self._w   = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._buf        = []
        self._last_flush = time.monotonic()

    def log(self, **row):
        self._buf.append({k: row.get(k, "") for k in self.FIELDS})
        if len(self._buf) >= 50 or (time.monotonic() - self._last_flush) >= 0.5:
            self._flush()

    def _flush(self):
        if not self._buf:
            return
        self._w.writerows(self._buf)
        self._buf.clear()
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self):
        try:
            self._flush()
        finally:
            self._f.close()


# ─── Encoder ──────────────────────────────────────────────────────────────────
_bus = smbus.SMBus(1)

def read_raw_angle() -> float:
    data = _bus.read_i2c_block_data(0x40, 0xFE, 2)
    return data[0] / 255.0 + data[1] / 64.0 / 255.0


# ─── Helpers ──────────────────────────────────────────────────────────────────
async def wait_for_enter(prompt: str = "Press ENTER to start..."):
    await asyncio.to_thread(input, prompt)

async def get_odrive_velocity_rad_s(odrive) -> float:
    if   hasattr(odrive, "get_velocity"):  v = await odrive.get_velocity()
    elif hasattr(odrive, "velocity"):      v = odrive.velocity
    elif hasattr(odrive, "vel_estimate"):  v = odrive.vel_estimate
    else:                                  v = 0.0
    return float(v) * 2.0 * np.pi


# ─── Controller ───────────────────────────────────────────────────────────────
async def controller(odrive):
    await asyncio.sleep(0.002)
    odrive.set_controller_mode("torque_control")

    loop      = asyncio.get_running_loop()
    stop_at   = datetime.now() + timedelta(seconds=RUN_DURATION_S)
    next_time = loop.time()

    rls    = RLS1(lambda_f=RLS_LAMBDA, p0=RLS_P0, theta_init=A_HAT_INIT)
    ekf    = PendulumEKF(I_PENDULUM, EKF_A)   # a=0: pure smoother, no gravity
    u      = 0.0

    vdot_lp = LowPass1(fc_hz=VDOT_LP_FC_HZ)   # smooth raw accel for RLS

    # Auto-trim state
    vel_lp       = LowPass1(fc_hz=TRIM_LP_FC_HZ)
    eq_trim_rad  = 0.0
    eq_trim_rad1 = 0.0
    eq_trim_deg  = 0.0
    trim_max_rad = TRIM_MAX_DEG * np.pi / 180.0

    # Calibration
    rest_pos = read_raw_angle()
    print("\nSystem initialized. Move pendulum near upright.")
    await wait_for_enter("Press ENTER to start control... ")

    t0           = loop.time()
    log_period_s = 1.0 / LOG_FREQ_HZ
    last_log_s   = 0.0
    logger       = CsvLogger(prefix="pendulum_control")
    print(f"[CSV] Logging to: {logger.path}")

    val    = read_raw_angle()
    hturns = 0
    odrive.set_torque(0.0)

    t_prev = loop.time()
    v_prev = 0.0           # for raw finite-difference accel

    pi    = np.pi
    omega = TRAJ_FREQ_HZ * 2.0 * pi
    A     = (TRAJ_AMP_DEG / 180.0) * pi

    try:
        while datetime.now() < stop_at:
            next_time += DT_TARGET
            await asyncio.sleep(max(0.0, next_time - loop.time()))

            # Encoder unwrap
            prev_val = val
            val      = read_raw_angle()
            diff     = val - prev_val
            if   diff < -0.5: hturns += 1
            elif diff >  0.5: hturns -= 1

            p_meas = (val + hturns - rest_pos) * pi
            p      = p_meas - eq_trim_rad1

            # Timing
            t_now  = loop.time()
            dt     = t_now - t_prev
            t_prev = t_now
            if dt <= 0 or dt > 0.05:
                dt = DT_TARGET
            t_s = t_now - t0

            # Reference trajectory
            SIN_Wt = np.sin(-omega * t_s)
            x_d    =  A * SIN_Wt
            xd_d   = -A * omega   * np.cos(-omega * t_s)
            xdd_d  = -A * omega**2 * SIN_Wt

            # EKF — pure smoother (a=0), gives clean p_hat and v
            p_hat, v = ekf.update(p, u, dt)
            e  = p - x_d
            ed = v - xd_d
            s  = ed + C_SLIDE * e

            # Raw finite-difference acceleration (filtered) for RLS residual
            # I*v_dot = u - a*sin(p)  =>  d = u - I*v_dot = a*sin(p)
            vdot_raw = (v - v_prev) / dt
            vdot_filt = vdot_lp.update(vdot_raw, dt)
            d_rls    = u - I_PENDULUM * vdot_filt   # ← actual gravity torque signal
            v_prev   = v

            # ODrive velocity + auto-trim
            phi      = await get_odrive_velocity_rad_s(odrive)
            phi_filt = vel_lp.update(phi, dt)

            steady = (abs(v) < 0.5) and (abs(s) < 0.5)
            if steady and abs(phi_filt) > TRIM_DEADBAND:
                dtrim     = -TRIM_K_I * phi_filt * dt
                dtrim_max = (TRIM_SLEW_DEG_S * pi / 180.0) * dt
                dtrim     = float(np.clip(dtrim, -dtrim_max, dtrim_max))
                eq_trim_rad  += dtrim
                eq_trim_rad1  = TRIM_SWITCH * (eq_trim_rad - TRIM_KP * phi_filt)
                eq_trim_rad1  = float(np.clip(eq_trim_rad1, -trim_max_rad, trim_max_rad))

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # RLS update — regressor sin(p), target d = a*sin(p)
            phi_rls          = np.sin(p)
            a_hat_raw, rls_err = rls.update(phi_rls, d_rls)

            # Clamp only for control — never write back to rls.theta
            a_hat = float(np.clip(a_hat_raw, A_HAT_MIN, A_HAT_MAX))
            u_ad  = a_hat * phi_rls

            # Control law
            u = (
                - I_PENDULUM * xdd_d
                + K_RHO * s
                + K_S   * float(np.sign(s))
                + K_E   * e
                + I_PENDULUM * C_SLIDE * ed
                + u_ad
            )
            u = float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))
            odrive.set_torque(u)

            # Logging
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s, p_rad=p, v_rad_s=v,
                    x_d=x_d, xd_d=xd_d, e=e, ed=ed, rho=s,
                    a_hat=a_hat, a_hat_raw=a_hat_raw, rls_error=rls_err,
                    u_ad=u_ad, u_torque=u,
                    phi_rad_s=phi, phi_filt_rad_s=phi_filt,
                    eq_trim_deg=eq_trim_deg,
                    vdot_raw=vdot_raw, d_rls=d_rls,
                    dt_s=dt,
                )
                last_log_s = t_s

            print(
                f"t={t_s:6.2f}s | "
                f"a_hat={a_hat:+.4f} (raw={a_hat_raw:+.4f}) | "
                f"d_rls={d_rls:+.4f} | "
                f"rls_err={rls_err:+.4f} | "
                f"s={s:+.4f} | "
                f"u={u:+.4f} Nm | "
                f"p={p*180/pi:+.2f} deg"
            )

            await asyncio.sleep(0)

    finally:
        try:
            odrive.set_torque(0.0)
        except Exception:
            pass
        logger.close()
        print("[CSV] Logger closed.")


# ─── ODrive setup ─────────────────────────────────────────────────────────────
odrive = pyodrivecan.ODriveCAN(0)

async def main():
    odrive.clear_errors(identify=False)
    print("Cleared errors.")
    await asyncio.sleep(1)

    odrive.initCanBus()
    print("Place arm at bottom-center to calibrate zero position.")
    await asyncio.sleep(1)
    print(f"Encoder position: {odrive.position}")
    await asyncio.sleep(1)

    odrive.setAxisState("open_loop_control")

    await asyncio.gather(
        odrive.loop(),
        controller(odrive),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")
        try:
            odrive.set_torque(0.0)
        except Exception:
            pass
        try:
            odrive.estop()
        except Exception:
            pass
