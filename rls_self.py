import pyodrivecan
import asyncio
import numpy as np
from datetime import datetime, timedelta
import smbus
import time
from pathlib import Path
from PendulumEKF import PendulumEKF

import csv


# ─────────────────────────────────────────────────────────────────────────────
#  RECURSIVE LEAST SQUARES  (scalar parameter version)
#
#  Model:   y[k]  =  φ[k] · a  +  noise
#
#  where
#    φ[k] = sin(θ[k])          (regressor, scalar)
#    y[k] = torque residual     (see build_rls_output() below)
#    a    = mgl / I             (unknown scalar parameter)
#
#  Update equations  (with forgetting factor λ):
#    K[k]  = P[k-1]·φ[k] / (λ + φ[k]²·P[k-1])
#    a[k]  = a[k-1] + K[k]·(y[k] - φ[k]·a[k-1])
#    P[k]  = (1/λ)·(1 - K[k]·φ[k])·P[k-1]
#
#  P is the (scalar) covariance / "confidence" estimate.
#  Large P  → estimator moves quickly  (high uncertainty).
#  Small P  → estimator is locked in   (high confidence).
# ─────────────────────────────────────────────────────────────────────────────
class ScalarRLS:
    """
    Single-parameter recursive least squares with forgetting factor.

    Parameters
    ----------
    lam   : forgetting factor  (0 < λ ≤ 1).  λ=1 → no forgetting (standard
            RLS).  λ=0.98 is a good default for slowly time-varying params.
    P0    : initial covariance (scalar).  Large value = high initial uncertainty
            → estimator trusts data quickly.  Default 1000.
    a0    : initial parameter estimate.  Default 0.0.
    a_min, a_max : optional hard clamps on the estimate (safety guard).
    """

    def __init__(
        self,
        lam: float = .9990,
        P0: float = 11000.0,
        a0: float = 0.0,
        a_min: float = -50.0,
        a_max: float = 50.0,
    ):
        self.lam = float(lam)
        self.P = float(P0)
        self.a = float(a0)
        self.a_min = float(a_min)
        self.a_max = float(a_max)

        # Diagnostics (read-only after update)
        self.K = 0.0        # last Kalman gain
        self.innov = 0.0    # last innovation  y - φ·a_prev

    def update(self, phi: float, y: float) -> float:
        """
        Run one RLS step.

        Parameters
        ----------
        phi : regressor value  φ[k]  (= sin(θ) in our case)
        y   : output / measurement  y[k]  (torque residual)

        Returns
        -------
        Updated parameter estimate  a[k]
        """
        # Kalman gain
        denom = self.lam + phi * phi * self.P
        if abs(denom) < 1e-12:          # numerical guard
            return self.a

        self.K = self.P * phi / denom

        # Innovation
        self.innov = y - phi * self.a

        # Parameter update
        self.a = self.a + self.K * self.innov

        # Clamp  (prevents wind-up if the regressor is persistently small)
        self.a = max(self.a_min, min(self.a_max, self.a))

        # Covariance update
        self.P = (1.0 / self.lam) * (1.0 - self.K * phi) * self.P

        # Keep P from collapsing to zero (numerical floor)
        if self.P < 1e-6:
            self.P = 1e-6

        return self.a


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

        # ── Identical column layout to the original script ──────────────────
        # fc_est and fv_est are always 0.0 (not estimated here) but kept so
        # your existing post-processing scripts don't need changes.
        self.fieldnames = fieldnames or [
            "t_s",              # time since calibration (s)
            "p_turns",          # pendulum angle (turns)
            "p_rad",            # pendulum angle (rad)
            "v_turns_s",        # pendulum velocity (rad/s)
            "phi_rad_s",        # wheel speed (rad/s)
            "a_est",            # RLS parameter estimate  (mgl/I)
            "u_torque",         # commanded torque
            "raw_turns",        # raw encoder turns [0..~1)
            "rest_turns",       # calibration rest position
            "hturns",           # wrap counter
            "dt_s",             # loop dt
            "x_d_s",            # desired state
            "rho_s",            # sliding surface value
            "SV_s",             # RLS covariance P  (replaces singular value)
            "accel_s",          # accel (I * p_ddot)
            "eq_trim_deg",      # AUTO TRIM (deg)
            "phi_filt_rad_s",   # filtered wheel speed used for trim
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
        if (
            len(self._buf) >= self._flush_every
            or (now - self._last_flush) >= self._flush_interval_s
        ):
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


################# WAIT #################
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
        self.x_prev = float(x)
        return self.dx


class LowPass1:
    """1st-order low-pass: y <- y + alpha*(x - y), alpha = dt/(tau+dt)."""

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
    """
    Full 3-element regressor kept for compatibility, but only element [0]
    (sin θ) is used by the scalar RLS estimator.
    """
    v_esp = 0.1
    return np.array(
        [np.sin(theta), -0 * np.tanh(angular_velocity / v_esp), -0 * angular_velocity]
    )


################## RLS output signal ##################
def build_rls_output(
    u: float,
    I: float,
    xdd: float,
) -> float:
    """
    Isolate the part of the control law that  a·sin(θ)  must match.

    Control law (from original script):
        u = -xdd_d*I + k_rho*rho + k_s*sign(rho) + e + I*C*ed + u_ad - 0.00*phi

    Rearranging for  u_ad = a·φ[0] = a·sin(θ):
        a·sin(θ)  ≈  u + xdd_d*I - k_rho*rho - k_s*sign(rho) - e - I*C*ed

    The RLS "output" y is therefore the right-hand side above.
    Note: sign(rho) creates a discontinuous signal — if that injects noise into
    the estimator you can disable k_s below.
    """
    y = (
        -u
        + I * xdd
    )
    return float(y)


################## ODrive helpers ##################
async def get_odrive_velocity_rad_s(odrive) -> float:
    if hasattr(odrive, "get_velocity"):
        v = await odrive.get_velocity()
    elif hasattr(odrive, "velocity"):
        v = odrive.velocity
    elif hasattr(odrive, "vel_estimate"):
        v = odrive.vel_estimate
    else:
        v = 0.0
    return float(v) * 2.0 * np.pi


################## CONTROL LOOP ##################
async def controller(odrive):
    await asyncio.sleep(0.002)

    odrive.set_controller_mode("torque_control")
    stop_at = datetime.now() + timedelta(seconds=10000)
    next_time = 0
    dt_target = 0.001

    # ── RLS estimator ────────────────────────────────────────────────────────
    # λ = 0.98  →  effective memory ≈ 1/(1-λ) = 50 samples  (~50 ms at 1 kHz)
    # P0 = 1000 →  estimator trusts incoming data immediately
    # a0 = 1.006 → warm-start near your known value (same as a_actual above)
    rls = ScalarRLS(lam=0.98, P0=1000.0, a0=1.006, a_min=-20.0, a_max=20.0)

    # Gains  (unchanged from original)
    C      = 6
    k_rho  = 1
    k_s    = 0.0
    I      = 0.037322
    pi     = np.pi
    a_actual = 1.006

    ekf = PendulumEKF(I, a_actual)
    u = 0

    # ── AUTO TRIM ─────────────────────────────────────────────────────────────
    vel_lp = LowPass1(fc_hz=0.5)

    trim_switch      = 0
    eq_trim_rad      = 0.0
    eq_trim_rad1     = 0.0
    eq_trim_deg      = 0.0
    kp_trim          = 0.01
    trim_k_i         = 0.0001
    trim_deadband    = 0.001
    trim_max_deg     = 1.00
    trim_slew_deg_s  = 0.005

    # ── Calibration ──────────────────────────────────────────────────────────
    rest_pos = read_raw_angle()

    print("\nSystem initialized. Move pendulum near upright.")
    await wait_for_enter(
        "Press ENTER to start control (zero reference will be set now)... "
    )

    t0 = time.monotonic()

    # ── CSV log rate control ──────────────────────────────────────────────────
    log_frequency = 30
    log_period_s  = 1 / log_frequency
    last_log_s    = 0.0

    logger = CsvLogger(prefix="pendulum_rls")
    print(f"[CSV] Logging to: {logger.path}")

    # ── Unwrap state ──────────────────────────────────────────────────────────
    val    = read_raw_angle()
    hturns = 0

    odrive.set_torque(0)

    # ── Desired trajectory ────────────────────────────────────────────────────
    freq  = 0.8
    omega = freq * 2 * pi
    A_deg = 15
    A     = (A_deg / 180.0) * pi

    loop   = asyncio.get_running_loop()
    t_prev = loop.time()

    try:
        while datetime.now() < stop_at:
            next_time += dt_target
            await asyncio.sleep(max(0, next_time - loop.time()))

            # ── Encoder unwrap ────────────────────────────────────────────────
            prev_val = val
            val      = read_raw_angle()
            diff     = val - prev_val

            if diff < -0.5:
                hturns += 1
            elif diff > 0.5:
                hturns -= 1

            position = val + hturns
            p_meas   = (position - rest_pos) * pi
            p        = p_meas - eq_trim_rad1

            # ── Timing ───────────────────────────────────────────────────────
            t_now = loop.time()
            dt    = t_now - t_prev
            if dt <= 0:
                dt = 1e-6
            t_s = t_now - t0

            # ── Desired trajectory ────────────────────────────────────────────
            SIN_Wt = np.sin(-omega * t_s)
            offset = 0.0
            x_d    =  A * SIN_Wt + offset
            xd_d   = -A * omega * np.cos(-omega * t_s)
            xdd_d  = -A * omega * omega * SIN_Wt

            # ── State estimates ───────────────────────────────────────────────
            e = p - x_d
            p_hat, v = ekf.update(p, u, dt)

            theta_DD = ekf.dynamics(
                np.array([[p_hat], [v]]), u
            )[1, 0]

            ed  = v - xd_d
            rho = ed + C * e

            # ── ODrive velocity + AUTO TRIM ───────────────────────────────────
            phi_wheel = await get_odrive_velocity_rad_s(odrive)
            phi_filt  = vel_lp.update(phi_wheel, dt)

            steady = (abs(v) < 0.5) and (abs(rho) < 0.5)

            if steady and abs(phi_filt) > trim_deadband:
                dtrim = -trim_k_i * phi_filt * dt

                dtrim_max = (trim_slew_deg_s * pi / 180.0) * dt
                dtrim = max(-dtrim_max, min(dtrim_max, dtrim))

                eq_trim_rad  += dtrim
                eq_trim_rad1  = trim_switch * (eq_trim_rad - kp_trim * phi_filt)

                trim_max_rad = trim_max_deg * pi / 180.0
                eq_trim_rad1 = max(-trim_max_rad, min(trim_max_rad, eq_trim_rad1))

            eq_trim_deg = eq_trim_rad1 * 180.0 / pi

            # ── RLS update ────────────────────────────────────────────────────
            # Regressor: φ = sin(θ)  (scalar, element [0] of phi_regressor)
            # Note: we use x_d (desired angle) as regressor input, matching
            # the original script's  phi_regressor(x_d, v)  convention.
            phi_rls = np.sin(p)      # scalar regressor

            # Build the output signal y that a·sin(θ) must explain
            # (uses the u from the PREVIOUS time step — causal, no loop)
            y_rls = build_rls_output(
                u=u,
                I=I,
                xdd=theta_DD,
            )

            # Only update RLS while we have enough excitation and within
            # the first 10 s (mirrors the original SVD gating)
            #if t_s < 10.0:
            #    a_hat = rls.update(phi_rls, y_rls)
            #else:
            #    a_hat = rls.a   # freeze after 10 s
            a_hat = rls.update(phi_rls, y_rls)

            # ── Control law ──────────────────────────────────────────────────
            # u_ad = a_hat · sin(θ)   (scalar adaptive feed-forward)
            u_ad = a_hat * phi_rls

            u = (
                -xdd_d * I
                + k_rho * rho
                + k_s * np.sign(rho)
                + 1 * e
                + I * C * ed
                + u_ad
                - 0.00 * phi_wheel
            )

            odrive.set_torque(u)
            accel = I * theta_DD

            # ── Log (rate-limited) ────────────────────────────────────────────
            if (t_s - last_log_s) >= log_period_s:
                logger.log(
                    t_s=t_s,
                    p_turns=p / (2 * pi),
                    p_rad=p,
                    v_turns_s=v,
                    phi_rad_s=phi_wheel,
                    a_est=a_hat,          # RLS estimate of  mgl/I
                    u_torque=u,
                    raw_turns=val,
                    rest_turns=rest_pos,
                    hturns=hturns,
                    dt_s=dt,
                    x_d_s=x_d,
                    rho_s=rho,
                    SV_s=rls.P,           # log covariance P in the SV_s column
                    accel_s=accel,
                    eq_trim_deg=eq_trim_deg,
                    phi_filt_rad_s=phi_filt,
                )
                last_log_s = t_s

            t_prev = t_now

            print(
                f"a_hat = {a_hat:.4f},"
                f"  P = {rls.P:.4f},"
                f"  innov = {rls.innov:+.4f},"
                f"  p (deg) = {p * 180 / pi:.2f},"
                f"  trim (deg) = {eq_trim_deg:+.2f},"
                f"  phi_filt (rad/s) = {phi_filt:+.2f},"
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
