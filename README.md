# GPU-Based N-Body Simulation in Astrophysics

A high-performance GPU-accelerated N-body simulation framework built with Python, leveraging CuPy and Numba to model gravitational interactions between celestial bodies. This project enables fast computation and real-time visualization of astrophysical dynamics, making it suitable for studying stellar systems, galaxy formation, and other gravitational phenomena.

## Features

- **GPU Acceleration**: Utilizes CuPy and Numba CUDA JIT compilation for massive parallelization
- **Gravitational N-Body Simulation**: Models realistic gravitational interactions between particles
- **High Performance**: Capable of simulating thousands to millions of bodies in real-time
- **Visualization**: Built-in visualization tools for observing system evolution
- **Flexible Initial Conditions**: Support for various astronomical configurations (galaxies, star clusters, planetary systems)
- **Optimized Algorithms**: Implements efficient force calculation methods for large-scale simulations

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Simulation Parameters](#simulation-parameters)
- [Performance](#performance)
- [Examples](#examples)
- [Theory](#theory)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit (11.0 or later)
- Python 3.8 or higher
- Anaconda or Miniconda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/freyaabbi/GPU-Based-N-Body-Simulation-in-Astrophysics.git
cd GPU-Based-N-Body-Simulation-in-Astrophysics
```

2. Create a conda environment:
```bash
conda create -n nbody python=3.10
conda activate nbody
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy scipy matplotlib cupy-cuda11x numba
```

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
cupy-cuda11x>=11.0.0  # Use appropriate CUDA version
numba>=0.55.0
```

## Quick Start

Run a simple galaxy collision simulation:

```python
from nbody import Simulation, InitialConditions
import numpy as np

# Create two galaxies
n_particles = 10000
ic = InitialConditions()
galaxy1 = ic.create_galaxy(n_particles//2, mass=1e11, radius=10.0, position=[0, 0, 0])
galaxy2 = ic.create_galaxy(n_particles//2, mass=1e11, radius=10.0, position=[30, 0, 0])

# Combine systems
initial_state = ic.combine_systems([galaxy1, galaxy2])

# Create and run simulation
sim = Simulation(initial_state, dt=0.01, method='gpu')
sim.run(t_end=100.0, visualize=True)
```

## Usage

### Basic Simulation

```python
from nbody import Simulation
import numpy as np

# Define initial conditions
n_bodies = 1000
positions = np.random.randn(n_bodies, 3) * 10.0
velocities = np.random.randn(n_bodies, 3) * 0.5
masses = np.ones(n_bodies) * 1e30

# Initialize simulation
sim = Simulation(
    positions=positions,
    velocities=velocities,
    masses=masses,
    dt=0.01,
    softening=0.1,
    method='gpu'
)

# Run simulation
results = sim.run(t_end=100.0, save_interval=1.0)

# Visualize results
sim.animate(results, output='simulation.mp4')
```

### Custom Initial Conditions

```python
from nbody import InitialConditions

ic = InitialConditions()

# Solar system
solar_system = ic.solar_system()

# Star cluster
cluster = ic.plummer_sphere(n=5000, mass_total=1e38, radius=10.0)

# Galaxy
galaxy = ic.create_galaxy(n=10000, mass=1e42, radius=50.0)

# Custom configuration
custom = ic.from_arrays(positions, velocities, masses)
```

## Simulation Parameters

### Core Parameters

- **dt**: Timestep size (in simulation units)
- **softening**: Gravitational softening length (prevents singularities)
- **method**: Computation method (`'gpu'`, `'cpu'`, or `'auto'`)
- **integrator**: Time integration scheme (`'leapfrog'`, `'rk4'`, `'euler'`)

### Performance Tuning

```python
sim = Simulation(
    ...,
    block_size=256,      # CUDA block size
    use_shared_mem=True, # Enable shared memory optimization
    precision='float32'  # Or 'float64' for double precision
)
```

## Performance

Typical performance on NVIDIA RTX 3080 (10GB VRAM):

| Number of Bodies | FPS (Steps/sec) | Memory Usage |
|-----------------|----------------|--------------|
| 1,000           | ~500           | 100 MB       |
| 10,000          | ~150           | 200 MB       |
| 100,000         | ~8             | 1.5 GB       |
| 1,000,000       | ~0.3           | 12 GB        |

*Note: Performance varies based on GPU model, timestep, and simulation complexity*

## Examples

### 1. Galaxy Collision

```python
# Example of simulating a galaxy collision
from examples.galaxy_collision import run_galaxy_collision

run_galaxy_collision(
    n_particles_per_galaxy=5000,
    impact_parameter=20.0,
    relative_velocity=100.0,
    duration=200.0
)
```

### 2. Star Cluster Evolution

```python
# Simulate a globular cluster
from examples.star_cluster import run_cluster_evolution

run_cluster_evolution(
    n_stars=10000,
    cluster_radius=10.0,
    simulation_time=1000.0
)
```

### 3. Planetary System

```python
# Simulate a multi-planet system
from examples.planetary_system import create_system

system = create_system(n_planets=8, include_moons=True)
results = system.evolve(years=1000)
```

## Theory

### N-Body Problem

The N-body problem involves predicting the motion of N particles under mutual gravitational attraction. The gravitational force between particles i and j is:

```
F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
```

### Computational Complexity

- **Direct summation**: O(N²) per timestep
- **Tree methods** (future): O(N log N) per timestep
- **Fast Multipole Method** (future): O(N) per timestep

### Softening Parameter

To prevent numerical singularities when particles approach closely, a softening length ε is introduced:

```
F_ij = G * m_i * m_j * (r_j - r_i) / (|r_j - r_i|^2 + ε^2)^(3/2)
```

### Time Integration

The simulation supports multiple integration schemes:

- **Leapfrog (Kick-Drift-Kick)**: Symplectic, conserves energy well
- **Runge-Kutta 4th Order**: Higher accuracy, more computationally expensive
- **Euler**: Simple but less accurate (for testing only)

## Project Structure

```
GPU-Based-N-Body-Simulation-in-Astrophysics/
├── nbody/
│   ├── __init__.py
│   ├── simulation.py        # Main simulation class
│   ├── integrators.py       # Time integration methods
│   ├── initial_conditions.py # Setup functions
│   ├── force_gpu.py         # GPU force calculation
│   └── visualization.py     # Plotting and animation
├── examples/
│   ├── galaxy_collision.py
│   ├── star_cluster.py
│   └── planetary_system.py
├── tests/
│   ├── test_simulation.py
│   └── test_integrators.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 nbody/
black nbody/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by classical N-body simulation methods in astrophysics
- GPU acceleration techniques based on CUDA best practices
- CuPy and Numba teams for excellent GPU computing libraries
- Astrophysics community for validation data and test cases

## References

1. Dehnen, W. (2001). "Towards optimal softening in 3D N-body codes." MNRAS, 324, 273-291.
2. Aarseth, S. J. (2003). "Gravitational N-Body Simulations." Cambridge University Press.
3. Hockney, R. W., & Eastwood, J. W. (1988). "Computer Simulation Using Particles."
4. NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nbody_gpu_simulation,
  author = {Freya Abbi},
  title = {GPU-Based N-Body Simulation in Astrophysics},
  year = {2024},
  url = {https://github.com/freyaabbi/GPU-Based-N-Body-Simulation-in-Astrophysics}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Note**: This simulation is intended for educational and research purposes. For production astrophysical simulations, consider specialized codes like GADGET, RAMSES, or Nbody6++GPU.
