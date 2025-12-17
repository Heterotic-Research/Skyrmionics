//! Skyrmion: Spin dynamics utilities and LLGS simulator.
//!
//! This crate provides a small, self-contained implementation of the
//! Landau–Lifshitz–Gilbert–Slonczewski (LLGS) equation for single-spin
//! dynamics. It lets you integrate magnetization `m(t)` under an
//! arbitrary effective field via a 4th-order Runge–Kutta scheme with
//! per-step renormalization to keep `|m| ≈ 1`.
//!
//! Quick start:
//!
//! - Define initial magnetization and parameters.
//! - Provide an effective field function of time and magnetization.
//! - Call [`simulate_llgs`], then analyze the returned trajectory.
//!
//! Example
//! -------
//!
//! ```rust
//! use Skyrmion::{Vector3, LLGSParameters, simulate_llgs};
//!
//! let params = LLGSParameters { gyromagnetic_ratio: 1.0, damping: 0.1 };
//! let m0 = Vector3::new(1.0, 0.0, 0.0);
//! let result = simulate_llgs(m0, &params, 1e-3, 1.0, |t, _m| {
//!     // Constant field along +z; could depend on (t, m).
//!     Vector3::new(0.0, 0.0, 1.0)
//! });
//!
//! assert!((result.magnetization[0].magnitude() - 1.0).abs() < 1e-6);
//! ```
//!
//! Notes
//! -----
//! - Units are abstract: choose consistent scalings for `gamma`, `H`, and `t`.
//! - The integrator renormalizes after each step to limit drift.
//! - For multi-spin lattices and GPU acceleration, extend the field callback
//!   and parallelize the update; this crate focuses on single-spin clarity.
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
/// A simple 3-component vector.
///
/// Provides the basic operations needed for LLGS dynamics:
/// dot, cross, normalization, and scalar arithmetic.
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    /// Construct a new vector from components.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Dot product: `self · other`.
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product: `self × other`.
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Euclidean norm: `||self||`.
    pub fn magnitude(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit vector in the direction of `self`.
    ///
    /// Returns `self` unchanged if the magnitude is zero.
    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 { self } else { self / mag }
    }
}

impl core::ops::Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl core::ops::AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl core::ops::Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl core::ops::Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl core::ops::Mul<Vector3> for f64 {
    type Output = Vector3;

    fn mul(self, rhs: Vector3) -> Self::Output {
        rhs * self
    }
}

impl core::ops::Div<f64> for Vector3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl core::ops::MulAssign<f64> for Vector3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

#[derive(Clone, Copy, Debug)]
/// Parameters for the LLGS equation.
///
/// - `gyromagnetic_ratio` (`γ`): controls precession rate.
/// - `damping` (`α`): the Gilbert damping coefficient.
pub struct LLGSParameters {
    pub gyromagnetic_ratio: f64,
    pub damping: f64,
}

#[derive(Debug, Clone)]
/// Discrete trajectory of the simulation.
///
/// - `times`: the sampled times (starting at `t = 0`).
/// - `magnetization`: magnetization vector at each sampled time.
pub struct SimulationResult {
    pub times: Vec<f64>,
    pub magnetization: Vec<Vector3>,
}

/// Compute `dm/dt` for the LLGS equation at `(m, H_eff)`.
///
/// Uses the form:
///
/// $\displaystyle \frac{d\mathbf{m}}{dt} = \frac{-\gamma}{1+\alpha^2}
/// \left( \mathbf{m} \times \mathbf{H}_\text{eff} + \alpha\, \mathbf{m}
/// 	imes (\mathbf{m} \times \mathbf{H}_\text{eff}) \right)$
///
/// where `γ` is the gyromagnetic ratio and `α` is the damping.
pub fn llgs_derivative(m: Vector3, effective_field: Vector3, params: &LLGSParameters) -> Vector3 {
    let gamma = params.gyromagnetic_ratio;
    let alpha = params.damping;
    let mxh = m.cross(effective_field);
    let mxmxh = m.cross(mxh);
    let rhs = mxh + alpha * mxmxh;
    let prefactor = -gamma / (1.0 + alpha * alpha);
    prefactor * rhs
}

/// Integrate the LLGS equation using RK4 and per-step renormalization.
///
/// - `initial_magnetization`: starting `m(0)`; normalized internally.
/// - `params`: [`LLGSParameters`].
/// - `time_step`: fixed step size (must be positive).
/// - `total_time`: total simulation duration (must be positive).
/// - `effective_field`: callback providing `H_eff(t, m)`.
///
/// Returns a [`SimulationResult`] with sampled times and magnetization.
///
/// The intermediate RK stages are not renormalized to avoid biasing the
/// Butcher tableau; only the final update per step is normalized to reduce
/// drift in `|m|`.
pub fn simulate_llgs<F>(
    initial_magnetization: Vector3,
    params: &LLGSParameters,
    time_step: f64,
    total_time: f64,
    mut effective_field: F,
) -> SimulationResult
where
    F: FnMut(f64, Vector3) -> Vector3,
{
    assert!(time_step > 0.0, "time_step must be positive");
    assert!(total_time > 0.0, "total_time must be positive");

    let steps = (total_time / time_step).ceil() as usize;
    let mut times = Vec::with_capacity(steps + 1);
    let mut magnetization = Vec::with_capacity(steps + 1);

    let mut m = initial_magnetization.normalized();
    let mut t = 0.0;
    times.push(t);
    magnetization.push(m);

    for _ in 0..steps {
        let h1 = effective_field(t, m);
        let k1 = llgs_derivative(m, h1, params);

        let mid_step = time_step * 0.5;
        let m2 = m + k1 * mid_step;
        let h2 = effective_field(t + mid_step, m2.normalized());
        let k2 = llgs_derivative(m2, h2, params);

        let m3 = m + k2 * mid_step;
        let h3 = effective_field(t + mid_step, m3.normalized());
        let k3 = llgs_derivative(m3, h3, params);

        let m4 = m + k3 * time_step;
        let h4 = effective_field(t + time_step, m4.normalized());
        let k4 = llgs_derivative(m4, h4, params);

        // Runge-Kutta 4 integration with renormalization keeps |m| close to unity.
        let delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (time_step / 6.0);
        m = (m + delta).normalized();
        t += time_step;

        times.push(t);
        magnetization.push(m);
    }

    SimulationResult {
        times,
        magnetization,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_approx(lhs: Vector3, rhs: Vector3, tol: f64) {
        assert!((lhs.x - rhs.x).abs() <= tol, "x mismatch");
        assert!((lhs.y - rhs.y).abs() <= tol, "y mismatch");
        assert!((lhs.z - rhs.z).abs() <= tol, "z mismatch");
    }

    #[test]
    fn add_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn llgs_reduces_to_precession_without_damping() {
        let params = LLGSParameters {
            gyromagnetic_ratio: 1.0,
            damping: 0.0,
        };
        let m = Vector3::new(1.0, 0.0, 0.0);
        let h = Vector3::new(0.0, 0.0, 1.0);
        let rhs = llgs_derivative(m, h, &params);
        assert_vec3_approx(rhs, Vector3::new(0.0, 1.0, 0.0), 1e-12);
    }

    #[test]
    fn simulation_preserves_norm_without_damping() {
        let params = LLGSParameters {
            gyromagnetic_ratio: 1.0,
            damping: 0.0,
        };
        let initial = Vector3::new(1.0, 0.0, 0.0);
        let result = simulate_llgs(initial, &params, 1e-3, 0.1, |_, _| {
            Vector3::new(0.0, 0.0, 1.0)
        });

        for m in result.magnetization {
            let norm = m.magnitude();
            assert!(
                (norm - 1.0).abs() < 5e-4,
                "magnetization norm drifted: {norm}"
            );
        }
    }

    #[test]
    fn damping_aligns_spin_with_field() {
        let params = LLGSParameters {
            gyromagnetic_ratio: 1.0,
            damping: 0.5,
        };
        let initial = Vector3::new(0.0, 1.0, 0.0);
        let result = simulate_llgs(initial, &params, 1e-3, 10.0, |_, _| {
            Vector3::new(0.0, 0.0, 1.0)
        });

        let final_state = *result.magnetization.last().unwrap();
        assert!(
            final_state.z > 0.99,
            "spin failed to align with field: {final_state:?}"
        );
    }
}
