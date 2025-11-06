# GPU-Based N-Body Simulation in Astrophysics  
*A CUDA-accelerated Jupyter Notebook for simulating gravitational interactions between celestial bodies*  
by **Freya Abbi**

---

## Overview  
This repository presents a GPU-accelerated implementation of the N-Body problem within a Jupyter Notebook environment. It demonstrates how parallel computation using CUDA (via CuPy or Numba) can significantly improve the performance of astrophysical simulations compared to traditional CPU implementations.

The notebook walks through every step — from physics formulation and parameter initialization to visualization of celestial motion — making it both a learning resource and a foundation for further astrophysical research or high-performance computing experiments.

---

## Key Features  
- Fully self-contained Jupyter Notebook — run and visualize everything in one place.  
- GPU acceleration using CuPy or Numba for massively parallel gravitational force calculations.  
- Interactive visualization of particle motion and energy conservation over time.  
- Customizable parameters (number of bodies, time step, mass, softening factor).  
- Benchmark comparisons between CPU and GPU performance.  
- Educational clarity — step-by-step code cells illustrating core physics and algorithmic design.

---

## Theoretical Background  

The N-Body problem models the motion of *N* interacting particles under mutual gravitational forces:

\[
F_{ij} = G \frac{m_i m_j (r_j - r_i)}{(|r_j - r_i|^2 + \varepsilon^2)^{3/2}}
\]

where  
- \( F_{ij} \) is the gravitational force on body *i* due to body *j*,  
- \( G \) is the gravitational constant,  
- \( m_i, m_j \) are the masses,  
- \( r_i, r_j \) are position vectors, and  
- \( \varepsilon \) is the softening factor to prevent singularities.

Time integration is achieved using numerical schemes such as Euler or Leapfrog, enabling real-time system evolution.

---

## Requirements  

| Package | Purpose |
|----------|----------|
| numpy | CPU-based numerical operations |
| cupy or numba | GPU acceleration |
| matplotlib | Visualization and animation |
| tqdm | Progress bar for iterations |
| time | Runtime benchmarking |

### Installation  

Create a conda or virtual environment and install dependencies:

```bash
conda create -n nbody python=3.10
conda activate nbody
pip install numpy cupy-cuda11x matplotlib tqdm numba
