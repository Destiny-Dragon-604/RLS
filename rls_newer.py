"""
Pendulum Controller — Sliding Mode + 1-param RLS (estimates a = mgl)

v3 problem: EKF with a=0 is a pure integrator smoother — v_dot from it is
just double-differentiated smoothed encoder data, which is noisy garbage.

Fix: use a DirtyDerivative on raw p for velocity (same as working script 1),
and a second DirtyDerivative on that velocity for v_dot fed to RLS.
EKF is kept only for control error smoothing (p_hat, v_ekf for e, ed, s).

RLS signal chain:
    p  ->  DD(fc=5Hz)  ->  v_dd   (clean velocity)
    v_dd -> DD(fc=2Hz) ->  vdot   (clean accel, heavy filtering)
    d_rls = u_prev - I * vdot     (u_prev: one step delayed to avoid algebraic loop)
    phi_rls = sin(p)
    RLS fits: d_rls ≈ a_hat * sin(p)
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
EKF_A          =  0.50         # EKF is pure smoother only

TRAJ_FREQ_HZ   = 0.8
TRAJ_AMP_DEG   = 20.0

C_SLIDE        = 6.0
K_RHO          = 1.0
K_S            = 0.0
K_E            = 1.0

# RLS
RLS_LAMBDA     = 0.99
RLS_P0         = 100.0
A_HAT_INIT     = 0.4
A_HAT_MIN      = 0
A_HAT_MAX      = 3.0

# Dirty derivative cutoffs
VEL_DD_FC_HZ   = 5.0    # velocity from raw p  — higher = less lag, more noise
VDOT_DD_FC_HZ  = 1.0    # accel from velocity  — lower  = less noise for RLS

# Auto-trim
TRIM_SWITCH     = 0
TRIM_K_I        = 0.0001
TRIM_KP         = 0.01
TRIM_DEADBAND   = 0.001
TRIM_MAX_DEG    = 1.00
TRIM_SLEW_DEG_S = 0.005
TRIM_LP_FC_HZ   = 0.5

MAX_TORQUE     = 2.0
RUN_DURATION_S = 10000
LOG_FREQ_HZ    = 30


# ─── Filters ──────────────────────────────────────────────────────────────────
class DirtyDerivative:
    """
    Filtered derivative: dx = a*dx + (1-a)*(x-x_prev)/dt
    a = tau/(tau+dt), tau = 1/(2*pi*fc)
    """
    def __init__(self, fc_hz: float, x0: float = 0.0):
        self.tau  = 1.0 / (2.0 * np.pi * float(fc_hz))
        self.x_prev = float(x0)
        self.dx   = 0.0
        self.initialized = False

    def update(self, x: float, dt: float) -> float:
        if dt <= 0:
            return self.dx
        if not self.initialized:
            self.x_prev = float(x)
            self.dx     = 0.0
            self.initialized = True
            return self.dx
        a        = self.tau / (self.tau + dt)
        dx_raw   = (x - self.x_prev) / dt
        self.dx  = a * self.dx + (1.0 - a) * dx_raw
        self.x_prev = float(x)
        return self.dx


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
        self.theta = self.theta + K * error
        self.P     = (self.P - K * phi * self.P) / self.lambda_f
        return self.theta, error


# ─── CSV logger ───────────────────────────────────────────────────────────────
class CsvLogger:
    FIELDS = [
        "t_s", "p_rad", "v_dd", "v_ekf",
        "x_d", "e", "ed", "rho",
        "a_hat", "a_hat_raw", "rls_error",
        "d_rls", "phi_rls", "vdot",
        "u_ad", "u_torque",
        "phi_rad_s", "phi_filt_rad_s",
        "eq_trim_deg", "dt_s",
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

    rls = RLS1(lambda_f=RLS_LAMBDA, p0=RLS_P0, theta_init=A_HAT_INIT)
    ekf = PendulumEKF(I_PENDULUM, EKF_A)
    ekf1 = PendulumEKF(I_PENDULUM, .4)
    u   = 0.0
    u_prev = 0.0   # one-step delayed torque to avoid algebraic loop in d_rls

    # Dirty derivatives for RLS signal chain
    vel_dd  = DirtyDerivative(fc_hz=VEL_DD_FC_HZ)   # p  -> velocity
    vdot_dd = DirtyDerivative(fc_hz=VDOT_DD_FC_HZ)  # v  -> acceleration

    # Auto-trim
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

    pi    = np.pi
    omega = TRAJ_FREQ_HZ * 2.0 * pi
    A     = (TRAJ_AMP_DEG / 180.0) * pi

    try:
        while datetime.now() < stop_at:
            next_time += DT_TARGET
            await asyncio.sleep(max(0.0, next_time - loop.time()))

            # ── Encoder unwrap ───────────────────────────────────────────────
            prev_val = val
            val      = read_raw_angle()
            diff     = val - prev_val
            if   diff < -0.5: hturns += 1
            elif diff >  0.5: hturns -= 1

            p_meas = (val + hturns - rest_pos) * pi
            p      = p_meas - eq_trim_rad1
            p1  = p 

            # ── Timing ───────────────────────────────────────────────────────
            t_now  = loop.time()
            dt     = t_now - t_prev
            t_prev = t_now
            if dt <= 0 or dt > 0.05:
                dt = DT_TARGET
            t_s = t_now - t0

            # ── Reference trajectory ─────────────────────────────────────────
            SIN_Wt = np.sin(-omega * t_s)
            x_d    =  A * SIN_Wt
            xd_d   = -A * omega   * np.cos(-omega * t_s)
            xdd_d  = -A * omega**2 * SIN_Wt

            # ── Velocity from dirty derivative of raw p ───────────────────────
            # This is the clean signal — same approach as working script 1
            v_dd = vel_dd.update(p, dt)

            # ── EKF (pure smoother, used only for control errors) ────────────
            p_hat, v_ekf = ekf.update(p, u, dt)
            p_hat1, v_ekf1 = ekf1.update(p1,u,dt)

            # Use v_dd for control (less lag than EKF integrator with a=0)
            e  = p - x_d
            ed = v_dd - xd_d
            s  = ed + C_SLIDE * e

            # ── RLS signal: accel from dirty derivative of v_dd ──────────────
            #vdot = vdot_dd.update(v_dd, dt)
            vdot = ekf.dynamics(
                np.array([[p_hat1],[v_ekf1]]),
                u
                )[1,0]
            vdot1 = ekf.dynamics(
                np.array([[p_hat],[v_ekf]]),
                u
                )[1,0]

            # u_prev avoids algebraic loop: torque was set last step
            # I*vdot = u - a*sin(p)  =>  d = u - I*vdot = a*sin(p)
            d_rls   = u_prev - I_PENDULUM * vdot1
            phi_rls = np.sin(p)

            a_hat_raw, rls_err = rls.update(phi_rls, d_rls)

            # Clamp only for control output, never write back
            a_hat = float(np.clip(a_hat_raw, A_HAT_MIN, A_HAT_MAX))
            u_ad  = 0*a_hat * phi_rls

            # ── ODrive velocity + auto-trim ───────────────────────────────────
            phi      = await get_odrive_velocity_rad_s(odrive)
            phi_filt = vel_lp.update(phi, dt)

            steady = (abs(v_dd) < 0.5) and (abs(s) < 0.5)
            if steady and abs(phi_filt) > TRIM_DEADBAND:
                dtrim     = -TRIM_K_I * phi_filt * dt
                dtrim_max = (TRIM_SLEW_DEG_S * pi / 180.0) * dt
                dtrim     = float(np.clip(dtrim, -dtrim_max, dtrim_max))
                eq_trim_rad  += dtrim
                eq_trim_rad1  = TRIM_SWITCH * (eq_trim_rad - TRIM_KP * phi_filt)
                eq_trim_rad1  = float(np.clip(eq_trim_rad1, -trim_max_rad, trim_max_rad))

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # ── Control law ──────────────────────────────────────────────────
            u_prev = u
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
            
            ekf.update_parameters(a=a_hat)
            ekf1.update_parameters(a=a_hat)

            # ── Logging ──────────────────────────────────────────────────────
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s, p_rad=p, v_dd=v_dd, v_ekf=v_ekf,
                    x_d=x_d, e=e, ed=ed, rho=s,
                    a_hat=a_hat, a_hat_raw=a_hat_raw, rls_error=rls_err,
                    d_rls=d_rls, phi_rls=phi_rls, vdot=vdot,
                    u_ad=u_ad, u_torque=u,
                    phi_rad_s=phi, phi_filt_rad_s=phi_filt,
                    eq_trim_deg=eq_trim_deg, dt_s=dt,
                )
                last_log_s = t_s

            print(
                f"t={t_s:6.2f}s | "
                f"a={a_hat:+.4f}(raw={a_hat_raw:+.4f}) | "
                f"d={d_rls:+.5f} | "
                f"phi={phi_rls:+.4f} | "
                f"err={rls_err:+.5f} | "
                f"s={s:+.4f} | "
                f"u={u:+.4f} | "
                f"p={p*180/pi:+.2f}°"
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
