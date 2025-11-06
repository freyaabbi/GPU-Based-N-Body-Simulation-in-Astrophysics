GPU-Based N-Body Simulation in Astrophysics

A CUDA-accelerated Jupyter Notebook for simulating gravitational interactions between celestial bodies

🌌 Overview

This repository presents a GPU-accelerated implementation of the N-Body problem within a Jupyter Notebook environment. It demonstrates how parallel computation using CUDA (via CuPy or Numba) can significantly improve the performance of astrophysical simulations compared to traditional CPU implementations.

The notebook walks through every step — from physics formulation and parameter initialization to visualization of celestial motion — making it both a learning resource and a foundation for further astrophysical research or high-performance computing experiments.

🚀 Key Features

Fully self-contained Jupyter Notebook — run and visualize everything in one place.

GPU acceleration using CuPy or Numba for massively parallel gravitational force calculations.

Interactive visualization of particle motion and energy conservation over time.

Customizable parameters (number of bodies, time step, mass, softening factor).

Benchmark comparisons between CPU and GPU performance.

Educational clarity — step-by-step code cells illustrating core physics and algorithmic design.

🧠 Theoretical Background

The N-Body problem models the motion of N interacting particles under mutual gravitational forces:

𝐹
𝑖
𝑗
=
𝐺
𝑚
𝑖
𝑚
𝑗
(
𝑟
𝑗
−
𝑟
𝑖
)
(
∣
𝑟
𝑗
−
𝑟
𝑖
∣
2
+
𝜀
2
)
3
/
2
F
ij
	​

=G
(∣r
j
	​

−r
i
	​

∣
2
+ε
2
)
3/2
m
i
	​

m
j
	​

(r
j
	​

−r
i
	​

)
	​


where

𝐹
𝑖
𝑗
F
ij
	​

 is the gravitational force on body i due to body j,

𝐺
G is the gravitational constant,

𝑚
𝑖
,
𝑚
𝑗
m
i
	​

,m
j
	​

 are the masses,

𝑟
𝑖
,
𝑟
𝑗
r
i
	​

,r
j
	​

 are position vectors, and

𝜀
ε is the softening factor to prevent singularities.

Time integration is achieved using numerical schemes such as Euler or Leapfrog, enabling real-time system evolution.

⚙️ Requirements
Package	Purpose
numpy	CPU-based numerical operations
cupy or numba	GPU acceleration
matplotlib	Visualization and animation
tqdm	Progress bar for iterations
time	Runtime benchmarking
🧩 Installation

Create a conda or virtual environment and install dependencies:

conda create -n nbody python=3.10
conda activate nbody
pip install numpy cupy-cuda11x matplotlib tqdm numba


⚠️ Note: Ensure you have a CUDA-enabled NVIDIA GPU and drivers properly configured.

💻 Usage

Clone the repository

git clone https://github.com/freyaabbi/GPU-Based-N-Body-Simulation-in-Astrophysics.git
cd GPU-Based-N-Body-Simulation-in-Astrophysics


Launch Jupyter Notebook

jupyter notebook


Open the notebook file:
N Body Simulations in Astrophysics.ipynb

Run the notebook cells sequentially

The first few cells define parameters and helper functions.

The core cells perform GPU-based computation of forces and updates.

The final section visualizes particle trajectories and energy conservation.

⚡ Simulation Parameters
Parameter	Description	Example
N	Number of bodies	1024
G	Gravitational constant	6.674×10⁻¹¹
dt	Time step	0.01
softening	Softening parameter	0.1
iterations	Number of simulation steps	1000

You can edit these directly in the notebook to experiment with different systems (galaxies, clusters, binary stars, etc.).

📊 Performance & Benchmark
Mode	Device	Speedup over CPU
CPU (NumPy)	Intel i7	1× baseline
GPU (CuPy/Numba)	NVIDIA RTX 3050	~15–25× faster

Performance gain increases with number of bodies (N). Small systems may see less benefit due to kernel overhead.

🌠 Output & Visualization

The notebook generates:

Dynamic plots showing orbital motion and clustering behavior

Energy graphs verifying numerical stability

Animated trajectories (optional)

Timing summaries comparing CPU vs GPU runs

Example visualization (if added later):

from IPython.display import HTML
HTML(anim.to_html5_video())

🧩 Repository Structure
GPU-Based-N-Body-Simulation-in-Astrophysics/
├── N Body Simulations in Astrophysics.ipynb   # Main notebook (core project)
├── requirements.txt                           # Optional dependency list
└── README.md                                  # Project documentation

🔬 Educational Objective

This notebook aims to:

Demonstrate how GPU computing can accelerate physics simulations.

Provide a clear understanding of gravitational dynamics and numerical integration.

Serve as a teaching tool for computational astrophysics, physics, and high-performance computing courses.

📘 Future Improvements

Implement Barnes–Hut (O(N log N)) optimization for larger systems

Add real-time animation export (MP4/GIF)

Integrate energy conservation diagnostics

Explore multi-GPU scaling

🪐 References

Aarseth, S. J. Gravitational N-Body Simulations (Cambridge University Press, 2003).

Hockney & Eastwood, Computer Simulation Using Particles (1988).

NVIDIA, CUDA C Programming Guide.

Dehnen, W. (2001), MNRAS 324:273–291.
