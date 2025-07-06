import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
m_n = 1.67e-27
l_p = np.sqrt(hbar * G / c**3)
E_P = 1.22e19  # Planck energy (GeV)
Lambda = 1.1e-52
T_c = l_p / c
kappa_CTC = 0.813
kappa_J4 = 0.813
kappa_worm = 0.01
kappa_ent = 0.27
kappa_J6 = 0.01
kappa_J6_eff = 1e-33
kappa_ZPE = 0.1
kappa_grav = 1e-2
kappa_grav2 = 0.05
eta_CTC = 0.05
kappa_g = 0.1
beta_ZPE = 1e-3
lambda_r = 0.1  # Rainbow gravity parameter
d_c = 1e-9
lambda_v = 0.33333333326
beta = 1e-3
theta = 1.79
k = 1 / theta
omega = 2 * np.pi / (100 * 1e-12)
config = {'phase_shift': np.exp(1j * np.pi / 3), 'tetbit_scale': 1.0, 'scaling_factor': 1e-3, 'vertex_lambda': lambda_v}

# Lattice setup
nx, ny, nz, nt, nw1, nw2 = 5, 5, 5, 5, 3, 3
N = nx * ny * nz * nt * nw1 * nw2
dx = l_p * 1e5
dt = 1e-12
dw = l_p * 1e3
grid_shape = (nx, ny, nz, nt, nw1, nw2)

# Initialize state
psi = np.ones(grid_shape, dtype=np.complex128) / np.sqrt(N) * np.exp(1j * np.random.uniform(0, 2 * np.pi, grid_shape))
psi_past = psi.copy()
phi_N = np.random.uniform(-0.1, 0.1, (10, 10, 10))

class Tetbit:
    def __init__(self, config, position=None):
        self.config = config
        self.phase_shift = config["phase_shift"]
        self.state = np.zeros(4, dtype=np.complex128)
        self.state[0] = 1.0
        self.y_gate = self._tetrahedral_y()
        self.h_gate = self._tetrahedral_hadamard()
        if position is not None:
            self.encode_spacetime_position(position)

    def _tetrahedral_y(self):
        return np.array([[0, 0, 0, -self.phase_shift * 1j], [self.phase_shift * 1j, 0, 0, 0],
                         [0, self.phase_shift * 1j, 0, 0], [0, 0, self.phase_shift * 1j, 0]], dtype=np.complex128)

    def _tetrahedral_hadamard(self):
        phi = (1 + np.sqrt(5)) / 2
        scale = self.config['tetbit_scale'] * self.config['scaling_factor']
        h = np.array([[1, 1, 1, 1], [1, phi, -1/phi, -1], [1, -1/phi, phi, -1], [1, -1, -1, 1]], dtype=np.complex128) * scale
        norm = np.linalg.norm(h, axis=0)
        return h / norm[np.newaxis, :]

    def apply_gate(self, gate):
        self.state = gate @ self.state
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def measure(self):
        probs = np.abs(self.state)**2
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(4) / 4
        outcome = np.random.choice(4, p=probs)
        self.state = np.zeros(4, dtype=np.complex128)
        self.state[outcome] = 1.0
        return outcome

    def encode_spacetime_position(self, position):
        vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * self.config['vertex_lambda'] * self.config['scaling_factor']
        distances = np.linalg.norm(vertices - position[:3], axis=1)
        weights = np.exp(-distances**2 / (2 * self.config['tetbit_scale']**2))
        total_weight = np.sum(weights)
        if total_weight > 0:
            self.state = weights.astype(np.complex128) / np.sqrt(total_weight)
        return self.state

    def apply_wormhole_phase(self, u, v):
        phase = np.exp(1j * np.pi * (u * v) * self.config['scaling_factor'])
        self.state *= phase
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

class MetatronCircle:
    def __init__(self, config, center=np.array([0.0, 0.0, 0.0]), radius=1.0):
        self.config = config
        self.center = center
        self.radius = radius
        self.points = self._generate_points()
        self.phase_shift = config["phase_shift"]
        self.state = np.ones(len(self.points), dtype=np.complex128) / np.sqrt(len(self.points))

    def _generate_points(self):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = np.zeros_like(theta) + self.center[2]
        return np.column_stack((x, y, z))

    def apply_rotation(self, angle, axis):
        rotation_matrix = self._get_rotation_matrix(angle, axis)
        self.points = np.dot(self.points - self.center, rotation_matrix) + self.center
        self._update_state()

    def _get_rotation_matrix(self, angle, axis):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    def _update_state(self):
        distances = np.linalg.norm(self.points - self.center, axis=1)
        weights = np.exp(-distances**2 / (2 * self.config['tetbit_scale']**2))
        total_weight = np.sum(weights)
        if total_weight > 0:
            self.state = weights.astype(np.complex128) / np.sqrt(total_weight)
        else:
            self.state = np.ones(len(self.points), dtype=np.complex128) / np.sqrt(len(self.points))

    def apply_phase_shift(self):
        self.state *= self.phase_shift
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

def scalar_field(r_6d, t):
    return -r_6d**2 * np.cos(k * r_6d - omega * t) + 2 * r_6d * np.sin(k * r_6d - omega * t) + 2 * np.cos(k * r_6d - omega * t)

def nugget_field_deriv(t, phi_N_flat):
    phi_N = phi_N_flat.reshape((10, 10, 10))
    dphi_N_dt = np.zeros_like(phi_N)
    d2phi_N_dt2 = np.zeros_like(phi_N)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                laplacian = 0
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = (i+di)%10, (j+dj)%10, (k+dk)%10
                    laplacian += (phi_N[ni, nj, nk] - phi_N[i, j, k]) / dx**2
                laplacian -= 6 * phi_N[i, j, k] / dx**2
                nonlinear = phi_N[i, j, k] * (phi_N[i, j, k]**2 - 1) * (1 + 0.1 * np.sin(2 * np.pi * t))
                d2phi_N_dt2[i, j, k] = laplacian - 0.1**2 * phi_N[i, j, k] + 0.5 * phi_N[i, j, k] - nonlinear - (1/c**2) * dphi_N_dt[i, j, k]
    return np.concatenate([dphi_N_dt.flatten(), d2phi_N_dt2.flatten()])

def fractal_dimension_max(phi_N, r, t):
    grad_phi_N = np.zeros((10, 10, 10))
    for i in range(10):
        for j in range(10):
            for k in range(10):
                grad_x = (phi_N[(i+1)%10, j, k] - phi_N[(i-1)%10, j, k]) / (2 * dx)
                grad_y = (phi_N[i, (j+1)%10, k] - phi_N[i, (j-1)%10, k]) / (2 * dx)
                grad_z = (phi_N[i, j, (k+1)%10] - phi_N[i, j, (k-1)%10]) / (2 * dx)
                grad_phi_N[i, j, k] = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    d_f = 1.7 + 0.3 * np.tanh(np.abs(grad_phi_N)**2 / 0.1) + 0.05 * np.log(1 + r/l_p) * np.cos(2 * np.pi * t / T_c)
    multi_scale = sum(alpha * (r / (l_p * 10**(k-1)))**(d_f-3) * np.sin(2 * np.pi * t / (T_c * 10**(k-1)))
                      for k, alpha in enumerate([0.02, 0.01, 0.005], 1))
    return d_f + multi_scale

def zpe_density_max(r, phi, sigma, g_00, psi, psi_past, dx, t):
    rho_0 = -0.5 * hbar * c / d_c**4
    R_r = 1 + r/l_p + 0.2 * (r/l_p)**2
    A_phi = np.cos(phi)**2 + np.sin(phi)**2 + 0.1 * np.sin(2*phi)
    E_sigma = 1 + 0.15 * sigma / np.log2(N**3) * 1.618
    C_g = 1 + 0.05 * g_00 / (-c**2)
    F_r = -1.5e-17
    grad_psi = np.sum(np.abs(np.gradient(psi, dx))**2)
    N_phi_N = 1 + 0.2 * np.mean(np.abs(np.gradient(phi_N, dx))**2) * (r/l_p)**(d_f-3) * (1 + 0.1 * np.sin(2 * np.pi * np.log(r/l_p + 1e-10)))
    rho_zpe = rho_0 * R_r * A_phi * E_sigma * C_g * (1 + 0.1 * abs(F_r) / abs(-3.2e-17)) * N_phi_N
    ctc_feedback = eta_CTC * np.abs(psi_past)**2 / (np.abs(psi)**2 + 1e-10) * np.exp(-np.abs(np.angle(psi) - np.angle(psi_past)) / T_c)
    return rho_zpe * (1 + kappa_ZPE * grad_psi / (hbar**2 / d_c**2) + ctc_feedback)

def zpe_density_rainbow(r, phi, sigma, g_00, psi, psi_past, dx, t, E_P=1.22e19, lambda_r=0.1):
    zpe = zpe_density_max(r, phi, sigma, g_00, psi, psi_past, dx, t)
    E = hbar * omega * np.abs(psi)**2
    rainbow_factor = (1 + lambda_r * E / E_P)**(-1)
    return zpe * rainbow_factor

def rainbow_metric(r_6d, psi, psi_past, t, E_P=1.22e19, lambda_r=0.1):
    d_f = fractal_dimension_max(phi_N, r_6d, t)
    E = hbar * omega * np.abs(psi)**2
    rainbow_factor = (1 + lambda_r * E / E_P)**(-2)
    g_00 = -1 + 1e-5 * np.abs(psi)**2 * np.sin(k * r_6d - omega * t) + 1e-4 * d_f * np.abs(psi_past)**2 / (np.abs(psi)**2 + 1e-10) * np.exp(1j * T_c * np.tanh(np.angle(psi) - np.angle(psi_past)))
    return np.diag([g_00 * rainbow_factor, 1 + rainbow_factor, 1 + rainbow_factor, np.sinh(2 * r_6d)**2 + rainbow_factor])

def rainbow_qubit(psi, psi_past, r_6d, t, E_P=1.22e19, lambda_r=0.1):
    g_rainbow = rainbow_metric(r_6d, psi, psi_past, t, E_P, lambda_r)
    return np.sqrt(np.abs(g_rainbow[0,0])) * psi

def rainbow_gate(psi, psi_past, r_6d, t, E_P=1.22e19, lambda_r=0.1):
    g_rainbow = rainbow_metric(r_6d, psi, psi_past, t, E_P, lambda_r)
    J = np.abs(psi)**2 * np.linalg.inv(g_rainbow)
    tau = T_c * np.log(1 + t/T_c)
    U_g = np.exp(-1j * kappa_g / hbar * np.sum(g_rainbow * J, axis=(0,1)) * tau)
    zpe = zpe_density_rainbow(r_6d, phi, sigma, -1, psi, psi_past, dx, t, E_P, lambda_r)
    dV = dx**3 * dt * dw**2
    U_zpe = np.exp(1j * beta_ZPE * np.sum(zpe) * dV)
    return U_zpe * U_g * np.eye(N, dtype=np.complex128)

def hamiltonian_grav_ent_rainbow(psi, d_f, E_P=1.22e19, lambda_r=0.1):
    H_grav_ent = np.zeros_like(psi, dtype=np.complex128)
    coords = np.array(np.unravel_index(np.arange(N), grid_shape)).T
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    E = hbar * omega * np.abs(psi)**2
    rainbow_factor = (1 + lambda_r * E / E_P)**(-2)
    for idx in range(N):
        i, j, k, l, m, n = np.unravel_index(idx, grid_shape)
        sum_term = 0
        for idx2 in range(N):
            if idx2 != idx:
                i2, j2, k2, l2, m2, n2 = np.unravel_index(idx2, grid_shape)
                r_diff = np.sqrt((i-i2)**2 + (j-j2)**2 + (k-k2)**2 + 0.1*(l-l2)**2 + 0.1*(m-m2)**2 + 0.1*(n-n2)**2)
                sum_term += np.abs(psi[i2,j2,k2,l2,m2,n2])**2 * (1/r_diff**4 + d_f * np.exp(-r_diff/l_p)) / (r_diff**4 + 1e-10)
        H_grav_ent[i,j,k,l,m,n] = (kappa_grav + kappa_grav2 * d_f) * G * m_n * sum_term * psi[i,j,k,l,m,n] * rainbow_factor[idx]
    return H_grav_ent

def hamiltonian(psi, psi_past, phi_N, t, E_P=1.22e19, lambda_r=0.1):
    H_kin = np.zeros_like(psi, dtype=np.complex128)
    H_pot = np.zeros_like(psi, dtype=np.complex128)
    H_worm = np.zeros_like(psi, dtype=np.complex128)
    H_ent = np.zeros_like(psi, dtype=np.complex128)
    H_CTC = np.zeros_like(psi, dtype=np.complex128)
    H_J4 = np.zeros_like(psi, dtype=np.complex128)
    coords = np.array(np.unravel_index(np.arange(N), grid_shape)).T
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    phi = np.arctan2(coords[:,1], coords[:,0])
    sigma = 0.5 * np.log2(N**3)
    d_f = np.mean(fractal_dimension_max(phi_N, r_6d, t))
    for idx in range(N):
        i, j, k, l, m, n = np.unravel_index(idx, grid_shape)
        for d, delta in enumerate([dt, dx, dx, dx, dw, dw]):
            ni, nj, nk, nl, nm, nn = i, j, k, l, m, n
            if d == 0: ni = (i+1)%nx
            elif d == 1: nj = (j+1)%ny
            elif d == 2: nk = (k+1)%nz
            elif d == 3: nl = (l+1)%nt
            elif d == 4: nm = (m+1)%nw1
            elif d == 5: nn = (n+1)%nw2
            H_kin[idx] += -(hbar**2 / (2 * m_n)) * (psi[ni,nj,nk,nl,nm,nn] + psi[i%nx,j%ny,k%nz,l%nt,m%nw1,n%nw2] - 2*psi[i,j,k,l,m,n]) / delta**2
        V_grav = -G * m_n / (r_6d[idx]**4 * Lambda**2) * (1 + 1e-3 * (-np.sum(np.abs(psi)**2 * np.log(np.abs(psi)**2 + 1e-10))))
        H_pot[idx] = (V_grav * (1 + 2 * np.sin(t)) + 1e-2 * scalar_field(r_6d[idx], t)) * psi[i,j,k,l,m,n]
        H_worm[idx] = kappa_worm * np.exp(1j * 2 * t) * np.exp(-r_6d[idx]**2 / (2 * 1.0**2)) * psi[i,j,k,l,m,n]
        H_ent[idx] = sum(kappa_ent * (1 + np.sin(t)) * (psi[ni,nj,nk,nl,nm,nn] - psi[i,j,k,l,m,n]) * np.conj(psi[i%nx,j%ny,k%nz,l%nt,m%nw1,n%nw2] - psi[i,j,k,l,m,n])
                         for d, (ni, nj, nk, nl, nm, nn) in enumerate([(i+1,j,k,l,m,n), (i,j+1,k,l,m,n), (i,j,k+1,l,m,n), (i,j,k,l+1,m,n), (i,j,k,l,m+1,n), (i,j,k,l,m,n+1)]))
        H_CTC[idx] = kappa_CTC * np.exp(1j * T_c * np.tanh(np.angle(psi[i,j,k,l,m,n]) - np.angle(psi_past[i,j,k,l,m,n]))) * abs(psi[i,j,k,l,m,n])
        H_J4[idx] = kappa_J4 * np.sin(np.angle(psi[i,j,k,l,m,n])) * psi[i,j,k,l,m,n]
    return H_kin + H_pot + H_worm + H_ent + H_CTC + H_J4 + hamiltonian_grav_ent_rainbow(psi, d_f, E_P, lambda_r)

def plot_zpe_density(psi, phi_N, t, psi_past):
    coords = np.array(np.unravel_index(np.arange(N), grid_shape)).T
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    phi = np.arctan2(coords[:,1], coords[:,0])
    sigma = 0.5 * np.log2(N**3)
    zpe = zpe_density_rainbow(r_6d, phi, sigma, -1, psi, psi_past, dx, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=zpe, cmap='viridis')
    plt.colorbar(sc, label='Rainbow ZPE Density')
    plt.title(f'Rainbow ZPE Density at t={t:.2e} s')
    plt.show()

# Simulation
n_iterations = 100
t_span = (0, n_iterations * dt)
d_f_history = []
zpe_history = []
entanglement_history = []

for t_idx in range(n_iterations):
    t = t_idx * dt
    phi_N_flat = phi_N.flatten()
    sol = solve_ivp(nugget_field_deriv, [t, t + dt], np.concatenate([phi_N_flat, np.zeros_like(phi_N_flat)]), method='RK45')
    phi_N = sol.y[:phi_N.size, -1].reshape((10, 10, 10))
    
    coords = np.array(np.unravel_index(np.arange(N), grid_shape)).T
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    phi = np.arctan2(coords[:,1], coords[:,0])
    sigma = 0.5 * np.log2(N**3)
    
    d_f = fractal_dimension_max(phi_N, r_6d, t)
    d_f_history.append(np.mean(d_f))
    
    psi_g = rainbow_qubit(psi, psi_past, r_6d, t)
    U_g = rainbow_gate(psi, psi_past, r_6d, t)
    psi = U_g @ psi_g
    psi *= np.exp(1j * beta * scalar_field(r_6d, t))
    H = hamiltonian(psi, psi_past, phi_N, t)
    psi = psi - 1j * dt / hbar * H
    psi /= np.linalg.norm(psi)
    
    tet = Tetbit(config, position=np.array([0, 0, 0]))
    tet.apply_gate(tet.y_gate)
    for idx in range(N):
        if np.random.rand() < 0.1:
            psi[idx] *= tet.state[np.random.randint(4)]
    
    metatron = MetatronCircle(config)
    metatron.apply_phase_shift()
    psi *= metatron.state[np.random.randint(13)]
    
    zpe = zpe_density_rainbow(r_6d, phi, sigma, -1, psi, psi_past, dx, t)
    zpe_history.append(np.mean(zpe))
    
    probs = np.abs(psi)**2
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    entanglement_history.append(entropy)
    
    psi_past = psi.copy()

# Visualizations
plt.plot(np.arange(n_iterations), d_f_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Fractal Dimension')
plt.title('Fractal Dimension Evolution')
plt.show()

plt.plot(np.arange(n_iterations), zpe_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Rainbow ZPE Density')
plt.title('Rainbow ZPE Density Evolution')
plt.show()

plt.plot(np.arange(n_iterations), entanglement_history)
plt.xlabel('Iteration')
plt.ylabel('Entanglement Entropy')
plt.title('Entanglement Entropy Evolution')
plt.show()

vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * lambda_v
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='b')
plt.title('Tetrahedral Lattice')
plt.show()

plot_zpe_density(psi, phi_N, t, psi_past)