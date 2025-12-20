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

    /// Zero vector helper.
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
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

#[derive(Clone, Debug)]
/// Bulk material parameters for multilayer spintronic simulations.
///
/// The fields are dimensionless scalings following the reduced units used in
/// arXiv:2207.11606, letting callers inject the proper prefactors.
pub struct BulkMaterial {
    pub gyromagnetic_ratio: f64,
    pub damping: f64,
    pub saturation_magnetization: f64,
    pub exchange_stiffness: f64,
    pub anisotropy_axis: Vector3,
    pub anisotropy_constant: f64,
}

#[derive(Clone, Debug)]
/// Single macrospin layer within a multilayer stack.
pub struct MaterialLayer {
    pub material: BulkMaterial,
    pub thickness: f64,
}

#[derive(Clone, Debug)]
/// Coupling descriptor between two layers, including Rashba torques.
pub struct InterlayerCoupling {
    pub first: usize,
    pub second: usize,
    pub exchange_strength: f64,
    pub rashba_axis: Vector3,
    pub field_like_strength: f64,
    pub damping_like_strength: f64,
}

#[derive(Clone, Debug)]
/// Aggregates layers and interfacial couplings for bulk simulations.
pub struct MultilayerSystem {
    pub layers: Vec<MaterialLayer>,
    pub couplings: Vec<InterlayerCoupling>,
}

#[derive(Debug, Clone)]
/// Trajectory for all layers in a multilayer simulation.
pub struct MultilayerSimulationResult {
    pub times: Vec<f64>,
    pub magnetization: Vec<Vec<Vector3>>,
}

#[derive(Clone, Copy, Debug)]
/// Discrete lattice dimensions for bulk simulations.
pub struct LatticeDimensions {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl LatticeDimensions {
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

#[derive(Clone, Debug)]
/// Masks active sites within the lattice volume.
pub struct LatticeGeometry {
    pub dimensions: LatticeDimensions,
    pub cell_size: Vector3,
    pub mask: Vec<bool>,
}

impl LatticeGeometry {
    fn validate(&self) {
        let expected = self.dimensions.len();
        if !self.mask.is_empty() {
            assert_eq!(
                expected,
                self.mask.len(),
                "geometry mask must match lattice cell count",
            );
        }
    }

    fn is_active(&self, index: usize) -> bool {
        if self.mask.is_empty() {
            true
        } else {
            self.mask[index]
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Axis-specific boundary handling.
pub enum BoundaryCondition {
    Open,
    Periodic,
    Fixed(Vector3),
}

#[derive(Clone, Debug)]
pub struct BoundarySettings {
    pub x: BoundaryCondition,
    pub y: BoundaryCondition,
    pub z: BoundaryCondition,
}

impl BoundarySettings {
    fn for_axis(&self, axis: usize) -> BoundaryCondition {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// Directional Dzyaloshinskii-Moriya constants per axis.
pub struct DmiVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug, Default)]
/// Spin and charge current parameters for boundary-driven torques.
pub struct CurrentParameters {
    pub spin_polarization: Vector3,
    pub spin_current_density: Vector3,
    pub charge_current_density: Vector3,
    pub nonadiabatic_beta: f64,
    pub damping_like_coefficient: f64,
}

impl CurrentParameters {
    fn drift_velocity(&self) -> Vector3 {
        self.spin_current_density + self.charge_current_density
    }
}

#[derive(Clone, Debug)]
/// Configuration for bulk lattice simulations with DMI and currents.
pub struct BulkLatticeConfig {
    pub material: BulkMaterial,
    pub geometry: LatticeGeometry,
    pub boundary: BoundarySettings,
    pub dmi: DmiVector,
    pub current: Option<CurrentParameters>,
}

#[derive(Debug, Clone)]
/// Trajectory for lattice-resolved simulations.
pub struct LatticeSimulationResult {
    pub times: Vec<f64>,
    pub magnetization: Vec<Vec<Vector3>>,
}

#[derive(Clone, Copy, Debug)]
/// Monochromatic spin-wave drive descriptor.
pub struct SpinWaveSource {
    pub amplitude: f64,
    pub wave_vector: Vector3,
    pub frequency: f64,
    pub phase: f64,
    pub polarization: Vector3,
}

#[derive(Default)]
struct InterlayerContribution {
    exchange_field: Vector3,
    field_like_field: Vector3,
    damping_like_torque: Vector3,
}

const MIN_SCALE: f64 = 1e-12;

fn anisotropy_field(material: &BulkMaterial, m: Vector3) -> Vector3 {
    if material.anisotropy_constant == 0.0 || material.saturation_magnetization.abs() < MIN_SCALE {
        return Vector3::zero();
    }

    let axis = material.anisotropy_axis.normalized();
    let projection = m.dot(axis);
    let prefactor = 2.0 * material.anisotropy_constant
        / material
            .saturation_magnetization
            .abs()
            .max(MIN_SCALE);
    axis * (prefactor * projection)
}

fn interlayer_contribution(system: &MultilayerSystem, state: &[Vector3], index: usize) -> InterlayerContribution {
    let mut contribution = InterlayerContribution::default();
    let self_m = state[index];
    let layer = &system.layers[index];
    let material = &layer.material;
    let ms = material
        .saturation_magnetization
        .abs()
        .max(MIN_SCALE);
    let thickness = layer.thickness.abs().max(MIN_SCALE);

    for coupling in &system.couplings {
        let other_index = if coupling.first == index {
            coupling.second
        } else if coupling.second == index {
            coupling.first
        } else {
            continue;
        };

        let other_m = state[other_index];
        let axis = coupling.rashba_axis.normalized();
        let exchange_prefactor = coupling.exchange_strength * material.exchange_stiffness / (thickness * ms);
        contribution.exchange_field += (other_m - self_m) * exchange_prefactor;
        contribution.field_like_field += axis * coupling.field_like_strength;
        let damping_like = self_m.cross(self_m.cross(axis)) * coupling.damping_like_strength;
        contribution.damping_like_torque += damping_like;
    }

    contribution
}

fn add_scaled_state(base: &[Vector3], slope: &[Vector3], scale: f64) -> Vec<Vector3> {
    base.iter()
        .zip(slope.iter())
        .map(|(m, k)| *m + (*k * scale))
        .collect()
}

fn normalize_state(state: Vec<Vector3>) -> Vec<Vector3> {
    state.into_iter().map(Vector3::normalized).collect()
}

fn lattice_index(dims: &LatticeDimensions, x: usize, y: usize, z: usize) -> usize {
    (z * dims.ny + y) * dims.nx + x
}

fn unravel_index(index: usize, dims: &LatticeDimensions) -> (usize, usize, usize) {
    let x = index % dims.nx;
    let y = (index / dims.nx) % dims.ny;
    let z = index / (dims.nx * dims.ny);
    (x, y, z)
}

fn axis_spacing(geometry: &LatticeGeometry, axis: usize) -> f64 {
    match axis {
        0 => geometry.cell_size.x.abs().max(MIN_SCALE),
        1 => geometry.cell_size.y.abs().max(MIN_SCALE),
        2 => geometry.cell_size.z.abs().max(MIN_SCALE),
        _ => unreachable!(),
    }
}

fn boundary_neighbor_coords(
    dims: &LatticeDimensions,
    coords: (usize, usize, usize),
    axis: usize,
    offset: isize,
    boundary: BoundaryCondition,
) -> Option<(usize, usize, usize)> {
    let (nx, ny, nz) = (dims.nx as isize, dims.ny as isize, dims.nz as isize);
    let (mut xi, mut yi, mut zi) = (coords.0 as isize, coords.1 as isize, coords.2 as isize);

    match axis {
        0 => xi += offset,
        1 => yi += offset,
        2 => zi += offset,
        _ => unreachable!(),
    }

    if xi < 0 || xi >= nx || yi < 0 || yi >= ny || zi < 0 || zi >= nz {
        match boundary {
            BoundaryCondition::Open | BoundaryCondition::Fixed(_) => return None,
            BoundaryCondition::Periodic => {
                match axis {
                    0 => xi = (xi % nx + nx) % nx,
                    1 => yi = (yi % ny + ny) % ny,
                    2 => zi = (zi % nz + nz) % nz,
                    _ => unreachable!(),
                }
            }
        }
    }

    Some((xi as usize, yi as usize, zi as usize))
}

fn neighbor_state(
    config: &BulkLatticeConfig,
    state: &[Vector3],
    coords: (usize, usize, usize),
    axis: usize,
    offset: isize,
) -> Option<Vector3> {
    let boundary = config.boundary.for_axis(axis);
    match boundary_neighbor_coords(&config.geometry.dimensions, coords, axis, offset, boundary) {
        Some(neighbor_coords) => {
            let index = lattice_index(&config.geometry.dimensions, neighbor_coords.0, neighbor_coords.1, neighbor_coords.2);
            if config.geometry.is_active(index) {
                Some(state[index])
            } else {
                None
            }
        }
        None => match boundary {
            BoundaryCondition::Fixed(vec) => Some(vec.normalized()),
            _ => None,
        },
    }
}

fn axis_neighbors(
    config: &BulkLatticeConfig,
    state: &[Vector3],
    coords: (usize, usize, usize),
    axis: usize,
) -> (Option<Vector3>, Option<Vector3>) {
    let plus = neighbor_state(config, state, coords, axis, 1);
    let minus = neighbor_state(config, state, coords, axis, -1);
    (plus, minus)
}

fn gradient_along_axis(
    m: Vector3,
    plus: Option<Vector3>,
    minus: Option<Vector3>,
    spacing: f64,
) -> Vector3 {
    match (plus, minus) {
        (Some(p), Some(mn)) => (p - mn) / (2.0 * spacing),
        (Some(p), None) => (p - m) / spacing,
        (None, Some(mn)) => (m - mn) / spacing,
        _ => Vector3::zero(),
    }
}

fn lattice_derivative<F>(
    config: &BulkLatticeConfig,
    time: f64,
    state: &[Vector3],
    external_field: &mut F,
) -> Vec<Vector3>
where
    F: FnMut(f64, &[Vector3], usize, (usize, usize, usize)) -> Vector3,
{
    let geometry = &config.geometry;
    let dims = &geometry.dimensions;
    let material = &config.material;
    let params = LLGSParameters {
        gyromagnetic_ratio: material.gyromagnetic_ratio,
        damping: material.damping,
    };
    let ms = material.saturation_magnetization.abs().max(MIN_SCALE);
    let exchange = material.exchange_stiffness;

    let mut derivatives = Vec::with_capacity(state.len());

    for (index, &m) in state.iter().enumerate() {
        if !geometry.is_active(index) {
            derivatives.push(Vector3::zero());
            continue;
        }

        let coords = unravel_index(index, dims);
        let mut effective = external_field(time, state, index, coords);
        effective += anisotropy_field(material, m);

        let mut gradients = [Vector3::zero(); 3];

        for axis in 0..3 {
            let (plus, minus) = axis_neighbors(config, state, coords, axis);
            let spacing = axis_spacing(geometry, axis);

            if exchange.abs() > MIN_SCALE {
                let prefactor = 2.0 * exchange / (ms * spacing * spacing);
                if let Some(p) = plus {
                    effective += (p - m) * prefactor;
                }
                if let Some(n) = minus {
                    effective += (n - m) * prefactor;
                }
            }

            let grad = gradient_along_axis(m, plus, minus, spacing);
            gradients[axis] = grad;

            let dmi_strength = match axis {
                0 => config.dmi.x,
                1 => config.dmi.y,
                2 => config.dmi.z,
                _ => 0.0,
            };

            if dmi_strength.abs() > MIN_SCALE {
                let axis_vec = match axis {
                    0 => Vector3::new(1.0, 0.0, 0.0),
                    1 => Vector3::new(0.0, 1.0, 0.0),
                    2 => Vector3::new(0.0, 0.0, 1.0),
                    _ => unreachable!(),
                };
                effective += grad.cross(axis_vec) * dmi_strength;
            }
        }

        let mut derivative = llgs_derivative(m, effective, &params);

        if let Some(current) = &config.current {
            let drift = current.drift_velocity();
            let adiabatic = gradients[0] * drift.x + gradients[1] * drift.y + gradients[2] * drift.z;
            let mut torque = Vector3::zero();
            torque = torque - adiabatic;
            if current.nonadiabatic_beta.abs() > MIN_SCALE {
                torque += m.cross(adiabatic) * current.nonadiabatic_beta;
            }
            if current.damping_like_coefficient.abs() > MIN_SCALE {
                let polarization = current.spin_polarization.normalized();
                torque += m.cross(m.cross(polarization)) * current.damping_like_coefficient;
            }
            derivative += torque;
        }

        derivatives.push(derivative);
    }

    derivatives
}

/// Integrate a bulk lattice LLGS system with exchange, DMI, and current-driven torques.
pub fn simulate_lattice_llgs<F>(
    config: &BulkLatticeConfig,
    initial_magnetization: &[Vector3],
    time_step: f64,
    total_time: f64,
    mut external_field: F,
) -> LatticeSimulationResult
where
    F: FnMut(f64, &[Vector3], usize, (usize, usize, usize)) -> Vector3,
{
    assert!(time_step > 0.0, "time_step must be positive");
    assert!(total_time > 0.0, "total_time must be positive");

    let geometry = &config.geometry;
    geometry.validate();

    let cell_count = geometry.dimensions.len();
    assert_eq!(
        cell_count,
        initial_magnetization.len(),
        "initial lattice state length must match dimensions",
    );

    if !geometry.mask.is_empty() {
        for idx in geometry
            .mask
            .iter()
            .enumerate()
            .filter_map(|(i, active)| if *active { Some(i) } else { None })
        {
            let mag = initial_magnetization[idx].magnitude();
            assert!(mag > 0.0, "active cell {} must have non-zero magnetization", idx);
        }
    }

    let steps = (total_time / time_step).ceil() as usize;
    let mut times = Vec::with_capacity(steps + 1);
    let mut magnetization = Vec::with_capacity(steps + 1);
    let mut state: Vec<Vector3> = initial_magnetization
        .iter()
        .map(|m| m.normalized())
        .collect();

    times.push(0.0);
    magnetization.push(state.clone());

    let mut time = 0.0;

    for _ in 0..steps {
        let k1 = lattice_derivative(config, time, &state, &mut external_field);
        let state_k1 = normalize_state(add_scaled_state(&state, &k1, 0.5 * time_step));

        let k2 = lattice_derivative(
            config,
            time + 0.5 * time_step,
            &state_k1,
            &mut external_field,
        );
        let state_k2 = normalize_state(add_scaled_state(&state, &k2, 0.5 * time_step));

        let k3 = lattice_derivative(
            config,
            time + 0.5 * time_step,
            &state_k2,
            &mut external_field,
        );
        let state_k3 = normalize_state(add_scaled_state(&state, &k3, time_step));

        let k4 = lattice_derivative(
            config,
            time + time_step,
            &state_k3,
            &mut external_field,
        );

        let mut new_state = Vec::with_capacity(cell_count);
        for idx in 0..cell_count {
            let delta = (
                k1[idx]
                    + 2.0 * k2[idx]
                    + 2.0 * k3[idx]
                    + k4[idx]
            ) * (time_step / 6.0);
            new_state.push((state[idx] + delta).normalized());
        }

        time += time_step;
        state = new_state;

        times.push(time);
        magnetization.push(state.clone());
    }

    LatticeSimulationResult { times, magnetization }
}

/// Create a convenience spin-wave drive closure for lattice simulations.
pub fn spin_wave_drive<'a>(
    source: SpinWaveSource,
    geometry: &'a LatticeGeometry,
) -> impl FnMut(f64, &[Vector3], usize, (usize, usize, usize)) -> Vector3 + 'a {
    let polarization = source.polarization.normalized();
    let cell_size = geometry.cell_size;
    move |time, _state, _index, coords| {
        let position = Vector3::new(
            coords.0 as f64 * cell_size.x,
            coords.1 as f64 * cell_size.y,
            coords.2 as f64 * cell_size.z,
        );
        let phase = source.wave_vector.dot(position) - source.frequency * time + source.phase;
        let envelope = (phase).cos();
        polarization * (source.amplitude * envelope)
    }
}

fn multilayer_derivative<F>(
    system: &MultilayerSystem,
    time: f64,
    state: &[Vector3],
    external_field: &mut F,
) -> Vec<Vector3>
where
    F: FnMut(f64, &[Vector3], usize) -> Vector3,
{
    let mut derivatives = Vec::with_capacity(state.len());

    for (index, layer) in system.layers.iter().enumerate() {
        let material = &layer.material;
        let params = LLGSParameters {
            gyromagnetic_ratio: material.gyromagnetic_ratio,
            damping: material.damping,
        };

        let mut effective = external_field(time, state, index);
        effective += anisotropy_field(material, state[index]);

        let contribution = interlayer_contribution(system, state, index);
        effective += contribution.exchange_field + contribution.field_like_field;

        let mut derivative = llgs_derivative(state[index], effective, &params);
        derivative += contribution.damping_like_torque;
        derivatives.push(derivative);
    }

    derivatives
}

/// Simulate a multilayer macrospin stack with interfacial Rashba torques.
///
/// The coupling model follows the effective field construction used in
/// arXiv:2207.11606, combining bulk LLGS dynamics with interfacial exchange
/// and spin-orbit torques. Callers provide the external field term per layer
/// while this routine injects anisotropy, exchange, and Rashba contributions.
pub fn simulate_multilayer_llgs<F>(
    system: &MultilayerSystem,
    initial_magnetization: &[Vector3],
    time_step: f64,
    total_time: f64,
    mut external_field: F,
) -> MultilayerSimulationResult
where
    F: FnMut(f64, &[Vector3], usize) -> Vector3,
{
    assert!(time_step > 0.0, "time_step must be positive");
    assert!(total_time > 0.0, "total_time must be positive");
    assert!(!system.layers.is_empty(), "system must contain at least one layer");
    assert_eq!(
        system.layers.len(),
        initial_magnetization.len(),
        "initial state length must match layer count",
    );

    for coupling in &system.couplings {
        assert!(coupling.first < system.layers.len(), "invalid coupling index: first");
        assert!(coupling.second < system.layers.len(), "invalid coupling index: second");
        assert!(coupling.first != coupling.second, "degenerate coupling indices");
    }

    let steps = (total_time / time_step).ceil() as usize;
    let mut times = Vec::with_capacity(steps + 1);
    let mut magnetization = Vec::with_capacity(steps + 1);
    let mut state: Vec<Vector3> = initial_magnetization
        .iter()
        .map(|m| m.normalized())
        .collect();

    times.push(0.0);
    magnetization.push(state.clone());

    let mut time = 0.0;
    let layer_count = system.layers.len();

    for _ in 0..steps {
        let k1 = multilayer_derivative(system, time, &state, &mut external_field);
        let state_k1 = normalize_state(add_scaled_state(&state, &k1, 0.5 * time_step));

        let k2 = multilayer_derivative(
            system,
            time + 0.5 * time_step,
            &state_k1,
            &mut external_field,
        );
        let state_k2 = normalize_state(add_scaled_state(&state, &k2, 0.5 * time_step));

        let k3 = multilayer_derivative(
            system,
            time + 0.5 * time_step,
            &state_k2,
            &mut external_field,
        );
        let state_k3 = normalize_state(add_scaled_state(&state, &k3, time_step));

        let k4 = multilayer_derivative(
            system,
            time + time_step,
            &state_k3,
            &mut external_field,
        );

        let mut new_state = Vec::with_capacity(layer_count);
        for idx in 0..layer_count {
            let delta = (
                k1[idx]
                    + 2.0 * k2[idx]
                    + 2.0 * k3[idx]
                    + k4[idx]
            ) * (time_step / 6.0);
            new_state.push((state[idx] + delta).normalized());
        }

        time += time_step;
        state = new_state;

        times.push(time);
        magnetization.push(state.clone());
    }

    MultilayerSimulationResult { times, magnetization }
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
    use std::f64::consts::PI;

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

    #[test]
    fn multilayer_exchange_and_rashba_act() {
        let material = BulkMaterial {
            gyromagnetic_ratio: 1.0,
            damping: 0.1,
            saturation_magnetization: 1.0,
            exchange_stiffness: 1.0,
            anisotropy_axis: Vector3::new(0.0, 0.0, 1.0),
            anisotropy_constant: 0.0,
        };

        let system = MultilayerSystem {
            layers: vec![
                MaterialLayer {
                    material: material.clone(),
                    thickness: 1.0,
                },
                MaterialLayer {
                    material: material.clone(),
                    thickness: 1.0,
                },
            ],
            couplings: vec![InterlayerCoupling {
                first: 0,
                second: 1,
                exchange_strength: 0.5,
                rashba_axis: Vector3::new(0.0, 1.0, 0.0),
                field_like_strength: 0.2,
                damping_like_strength: 0.1,
            }],
        };

        let initial = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)];
        let result = simulate_multilayer_llgs(&system, &initial, 1e-3, 0.2, |_, _, _| {
            Vector3::zero()
        });

        assert_eq!(result.times.len(), result.magnetization.len());
        let final_layer0 = result.magnetization.last().unwrap()[0];
        let final_layer1 = result.magnetization.last().unwrap()[1];

        let initial_dot = initial[0].dot(initial[1]);
        let final_dot = final_layer0.dot(final_layer1);
        assert!(
            final_dot > initial_dot,
            "layers failed to approach alignment: initial {initial_dot}, final {final_dot}"
        );

        for layer_state in result.magnetization.last().unwrap() {
            let norm = layer_state.magnitude();
            assert!((norm - 1.0).abs() < 1e-6, "layer norm drifted: {norm}");
        }
    }

    #[test]
    fn lattice_with_dmi_preserves_norms() {
        let material = BulkMaterial {
            gyromagnetic_ratio: 1.0,
            damping: 0.05,
            saturation_magnetization: 1.0,
            exchange_stiffness: 1.0,
            anisotropy_axis: Vector3::new(0.0, 0.0, 1.0),
            anisotropy_constant: 0.0,
        };

        let geometry = LatticeGeometry {
            dimensions: LatticeDimensions { nx: 4, ny: 4, nz: 1 },
            cell_size: Vector3::new(1.0, 1.0, 1.0),
            mask: Vec::new(),
        };

        let config = BulkLatticeConfig {
            material,
            geometry,
            boundary: BoundarySettings {
                x: BoundaryCondition::Periodic,
                y: BoundaryCondition::Periodic,
                z: BoundaryCondition::Open,
            },
            dmi: DmiVector { x: 0.4, y: 0.4, z: 0.0 },
            current: None,
        };

        let dims = config.geometry.dimensions;
        let mut initial = Vec::with_capacity(dims.len());
        for y in 0..dims.ny {
            for x in 0..dims.nx {
                let angle = 2.0 * PI * (x as f64) / (dims.nx as f64);
                let tilt = 0.1 * (y as f64);
                initial.push(Vector3::new(angle.cos(), angle.sin(), tilt).normalized());
            }
        }

        let result = simulate_lattice_llgs(&config, &initial, 5e-4, 0.02, |_, _, _, _| {
            Vector3::zero()
        });

        assert_eq!(result.times.len(), result.magnetization.len());
        let final_state = result.magnetization.last().unwrap();

        for m in final_state {
            let norm = m.magnitude();
            assert!((norm - 1.0).abs() < 1e-6, "cell norm drifted: {norm}");
        }

        let evolved = final_state
            .iter()
            .zip(initial.iter())
            .any(|(a, b)| (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs() > 1e-5);
        assert!(evolved, "lattice state failed to evolve under DMI dynamics");
    }

    #[test]
    fn spin_wave_drive_generates_expected_phase() {
        let geometry = LatticeGeometry {
            dimensions: LatticeDimensions { nx: 2, ny: 1, nz: 1 },
            cell_size: Vector3::new(0.5, 0.5, 0.5),
            mask: Vec::new(),
        };

        let source = SpinWaveSource {
            amplitude: 2.0,
            wave_vector: Vector3::new(PI, 0.0, 0.0),
            frequency: PI,
            phase: 0.0,
            polarization: Vector3::new(0.0, 1.0, 0.0),
        };

        let mut drive = spin_wave_drive(source, &geometry);
        let state = vec![Vector3::new(1.0, 0.0, 0.0); geometry.dimensions.len()];
        let field = drive(0.0, &state, 0, (0, 0, 0));
        let expected = Vector3::new(0.0, 2.0, 0.0);
        assert_vec3_approx(field, expected, 1e-6);
    }
}
