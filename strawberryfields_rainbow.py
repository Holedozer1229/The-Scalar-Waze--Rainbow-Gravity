import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from strawberryfields.remote import RemoteEngine

# Constants
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
m_n = 1.67e-27
l_p = np.sqrt(hbar * G / c**3)
E_P = 1.22e19  # Planck energy (GeV)
lambda_r = 0.1  # Rainbow gravity parameter
dx = l_p * 1e5
dt = 1e-13  # Reduced timestep for precision
omega = 2 * np.pi / (100 * 1e-12)
config = {'phase_shift': np.exp(1j * np.pi / 3), 'tetbit_scale': 1.0, 'scaling_factor': 1e-3, 'vertex_lambda': 0.33333333326}

# Simulation parameters
N_modes = 128  # Increased modes per chunk for better resolution
n_chunks = 2048 // N_modes  # 16 chunks for 2048 modes
n_iterations = 200  # Increased iterations for detailed evolution
cutoff_dim = 10  # Fock backend cutoff for precision
t_span = (0, n_iterations * dt)

# Initialize remote engine for scalability (Xanadu cloud)
try:
    eng = RemoteEngine("X8", device_id="X8-1")  # Example X8 device, adjust as needed
except Exception:
    print("Remote engine unavailable, falling back to local Fock engine")
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})

# Define CV geometric structures with enhanced precision
class TetbitCV:
    def __init__(self, config):
        self.config = config
        self.phase_shift = config["phase_shift"]

    def apply(self, q):
        Rgate(np.angle(self.phase_shift), 0.01) | q[0]  # Precise phase rotation with small angle

class MetatronCircleCV:
    def __init__(self, config):
        self.config = config
        self.phase_shift = config["phase_shift"]

    def apply(self, q):
        for i in range(0, N_modes, 2):  # Apply to every second mode for denser coverage
            Phase(np.angle(self.phase_shift) * (i / N_modes)) | q[i]  # Gradual phase shift

# Rainbow gravity Hamiltonian with higher-order terms
def rainbow_hamiltonian(q, r_6d, t):
    state = eng.run(prog).state
    E = hbar * omega * np.abs(state.mean_amplitude(q[0]))**2  # Energy from mean amplitude
    rainbow_factor = 1 / (1 + lambda_r * (E / E_P)**1.5)  # Nonlinear rainbow factor for precision
    V_grav = -G * m_n / (r_6d**4 + 1e-10) * rainbow_factor  # Avoid division by zero
    H = V_grav * (q[0].x**2 + q[0].p**2) + 0.01 * (q[0].x**4 + q[0].p**4)  # Add quartic terms
    return H

# Precise evolution with time-dependent operators
def apply_rainbow_evolution(prog, q, r_6d, t, dt):
    with prog.context as q:
        H = rainbow_hamiltonian(q, r_6d, t)
        # Time evolution via Hamiltonian simulation
        expH = (-1j * H * dt / hbar).eval()  # Exponential of Hamiltonian
        CustomOperator(expH) | q[0]  # Apply custom evolution
        Dgate(0.01 * np.sin(omega * t) * rainbow_factor) | q[0]  # Displacement
        Sgate(0.1 * rainbow_factor * np.cos(omega * t)) | q[0]  # Time-varying squeezing
    return prog

# Parallel execution of chunks
def simulate_chunk(chunk_idx):
    prog_chunk = sf.Program(N_modes)
    with prog_chunk.context as q:
        SqueezedX(0.1) | q[0]  # Initial state
        for i in range(N_modes-1):
            BSgate(np.pi/4, 0) | (q[i], q[i+1])  # Entanglement
    
    state_chunk = eng.run(prog_chunk).state
    zpe_history = []
    entanglement_history = []
    
    for t_idx in range(n_iterations):
        t = t_idx * dt
        coords = np.linspace(chunk_idx * l_p * 1e5, (chunk_idx + 1) * l_p * 1e5, N_modes)
        r_6d = np.sqrt(np.sum(coords**2 * [1, 1, 0.1, 0.1, 0.1, 0.1], axis=1))[0]  # First r_6d
        
        prog_chunk = apply_rainbow_evolution(prog_chunk, q, r_6d, t, dt)
        state_chunk = eng.run(prog_chunk).state
        
        # Precise ZPE density
        E = hbar * omega * np.abs(state_chunk.mean_amplitude(q[0]))**2
        zpe = -0.5 * hbar * c / (1e-9)**4 * (1 / (1 + lambda_r * E / E_P)) * (1 + 0.1 * np.sin(omega * t))
        zpe_history.append(np.mean(zpe))
        
        # Entropy via reduced density matrix
        entropy = state_chunk.entropy([0], [1]) if state_chunk.entropy([0], [1]) is not None else 0.0
        entanglement_history.append(entropy)
    
    # Apply geometric operations
    with prog_chunk.context as q:
        tetbit = TetbitCV(config)
        tetbit.apply(q)
        metatron = MetatronCircleCV(config)
        metatron.apply(q)
    state_chunk = eng.run(prog_chunk).state
    
    return zpe_history, entanglement_history

if __name__ == '__main__':
    # Parallel execution
    with mp.Pool(processes=min(n_chunks, mp.cpu_count())) as pool:
        results = pool.map(simulate_chunk, range(n_chunks))
    
    # Aggregate results
    zpe_history = np.concatenate([r[0] for r in results])
    entanglement_history = np.concatenate([r[1] for r in results])

    # Visualization
    time_points = np.arange(len(zpe_history)) * dt
    plt.figure(figsize=(12, 7))
    plt.plot(time_points, zpe_history)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Rainbow ZPE Density')
    plt.title('Rainbow ZPE Density Evolution')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(time_points, entanglement_history)
    plt.xlabel('Time (s)')
    plt.ylabel('Entanglement Entropy')
    plt.title('Entanglement Entropy Evolution')
    plt.grid(True)
    plt.show()

    # 3D Visualization of quadrature variances
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    x_var, p_var = eng.run(prog).state.quad_vacuum_var(0)
    ax.scatter([0], [0], [x_var], c='r', label='X Variance')
    ax.scatter([0], [0], [0], [p_var], c='b', label='P Variance')
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Variance')
    ax.set_title('Quadrature Variances at Initial State')
    ax.legend()
    plt.show()