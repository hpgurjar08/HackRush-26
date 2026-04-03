# Checkpoint 1 Writeup

## Title

**Benchmarking Machine-Learning and Classical Interatomic Potentials for Si and Ge**

## Note on Terminology

The project statement lists **MACE** as the compulsory MLIP. In this writeup, I assume the mention of `MASE` refers to **MACE**.

## 1. Brief Explanation of the Problem Statement

The problem asks us to study how well different atomistic models describe the interactions between atoms in **silicon (Si)** and **germanium (Ge)**. In molecular dynamics, atomic motion is obtained from Newton's equations, but the force on each atom must come from a model for the **potential energy** of the system. Once the energy is known, the force is obtained from the negative gradient of energy with respect to atomic position:

\[
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{R}_i}
\]

The assignment therefore reduces to a comparison of different approximations to the same potential energy surface.

The hackathon asks us to benchmark:

1. **Classical interatomic potentials** such as **Tersoff** and **ReaxFF**
2. **Machine-learning interatomic potentials (MLIPs)** such as **MACE** and **DeepMD**

and compare their predictions against **DFT and/or experimental reference data**.

The three target properties are:

1. **Lattice constant**
2. **Internal energy**
3. **Atomic forces**

These three properties probe different aspects of model quality:

- **Lattice constant** checks whether the model predicts the correct equilibrium crystal geometry
- **Internal energy** checks whether the model reproduces structural stability and the energy landscape
- **Forces** check whether the model reproduces the local driving forces governing atomistic motion

In short, the problem is a **benchmarking problem**, not just a simulation problem. We must justify which model is most accurate for Si and Ge and explain why.

## 2. Brief Methodology

The methodology should be designed so that each model is tested on exactly the same structural configurations.

### Step 1: Choose reference crystal structures

Start from the equilibrium **diamond-cubic** structures of Si and Ge.

### Step 2: Build a benchmark set

Use more than just the equilibrium cell.

- For lattice constant, perform structural relaxation
- For energy, generate strained cells around equilibrium and compare energy per atom
- For forces, generate distorted supercells with small random displacements

This is essential because a perfectly relaxed diamond structure has forces that are approximately zero by symmetry, which is not enough for a real force benchmark.

### Step 3: Obtain reference data

Use DFT-backed databases or DFT calculations.

- **Materials Project** for relaxed crystal structures and DFT energies
- **OQMD** for additional structure-energy data
- For forces, use force-labeled snapshots from a dataset or your own DFT calculations on distorted structures

### Step 4: Run each model on the same structures

For Tersoff, ReaxFF, MACE, and DeepMD, compute:

1. relaxed lattice constant
2. energy per atom
3. forces on distorted structures

### Step 5: Compare with metrics and plots

Use:

- lattice constant error
- energy MAE/RMSE
- force MAE/RMSE
- parity plots for energy and force

For lattice constant, a comparison table or bar chart is usually more informative than a parity plot because the number of data points is small.

## 3. What Are Interatomic Potentials and MLIPs?

## 3.1 Interatomic potentials

Interatomic potentials are analytical functions with fitted parameters. They approximate the energy of a material by combining physically motivated terms. They are usually much faster than DFT, but their accuracy is limited by the chosen functional form and parameterization.

### Tersoff

Tersoff is a **bond-order potential**. It contains repulsive and attractive radial terms, but the strength of a bond is modified by a bond-order term that depends on the local environment and bond angles. This makes it suitable for covalent materials such as Si and Ge.

### ReaxFF

ReaxFF is a **reactive force field**. It also uses bond-order ideas, but adds a larger number of interaction terms, including bond, angle, torsion, van der Waals, Coulomb, and charge-equilibration contributions. Because of this, it can model bond formation and bond breaking, but it is also more expensive and more sensitive to the parameter set used.

## 3.2 MLIPs

Machine-learning interatomic potentials learn the potential energy surface directly from data, usually DFT structures, energies, forces, and optionally stresses. Instead of imposing a fixed analytical functional form, they learn the mapping from local atomic environments to energy.

### MACE

MACE is an **equivariant message-passing neural network** for atomistic systems. It learns local atomic environments using symmetry-aware features and predicts the total energy as a sum of atomic contributions. Forces are obtained by differentiating the predicted energy with respect to positions.

### DeepMD

DeepMD represents each local atomic environment using symmetry-preserving descriptors and uses neural networks to map those descriptors to atomic energy contributions. The total energy is the sum over atoms, and forces are obtained from the energy gradient.

## 4. Properties to Predict: Physics and Calculation

## 4.1 Lattice constant

### What it means physically

The lattice constant is the equilibrium size of the crystal unit cell. For diamond-cubic Si and Ge, one parameter `a` defines the cubic cell.

If the model predicts the correct lattice constant, it means the balance between attractive and repulsive interactions is approximately correct near equilibrium.

### How to calculate it

There are two standard routes:

1. **Cell relaxation**
   relax atomic positions and cell vectors until forces and stress are minimized
2. **Energy-volume or energy-lattice scan**
   compute energy for several scaled cells and find the minimum

For cubic Si and Ge, after relaxation:

\[
a \approx l_x \approx l_y \approx l_z
\]

In LAMMPS, this is commonly done using `minimize` together with `fix box/relax`.

## 4.2 Internal energy

### What it means physically

Internal energy measures the stability of the atomic arrangement. In a ground-state structural benchmark, the most relevant quantity is usually the **potential energy**, because kinetic contributions are not the target of the model.

For same-composition comparisons, it is good practice to report:

1. **Energy per atom**
2. **Relative energy** with respect to the equilibrium configuration

If isolated-atom reference energies are available, one may also report **cohesive energy**:

\[
E_{\mathrm{coh}} = \frac{E_{\mathrm{bulk}}}{N} - E_{\mathrm{atom}}
\]

### How to calculate it

- relax the structure
- read the total potential energy `pe`
- divide by number of atoms for energy per atom
- optionally generate an energy-volume curve around equilibrium

For model comparison, relative energy curves are often more reliable than raw absolute energies.

## 4.3 Forces

### What they mean physically

Forces indicate how atoms will move and are therefore the most direct quantity for molecular dynamics. They are energy gradients:

\[
\mathbf{F}_i = - \nabla_i E
\]

### How to calculate them

For a given configuration, the model directly returns the force on every atom.

In LAMMPS, forces can be written using a custom dump:

```lammps
dump 1 all custom 1 forces.dump id type x y z fx fy fz
```

### Important methodological point

For a perfectly relaxed diamond-cubic Si or Ge crystal, the forces should be nearly zero. That means force benchmarking should **not** use only the equilibrium structure. Instead, use:

- isotropically strained cells
- supercells with small random displacements
- optionally snapshots from short finite-temperature runs

This gives a meaningful force distribution for parity plots and MAE/RMSE.

## 5. Model-by-Model Explanation and Workflow

## 5.1 Tersoff

### How it works

The LAMMPS documentation defines Tersoff as a 3-body bond-order potential. The total energy contains a repulsive radial term, an attractive radial term, and an environment-dependent bond-order term. This is why Tersoff captures angular dependence and works well for covalent materials.

Physically:

- two-body terms control attraction and repulsion with distance
- the bond-order term weakens or strengthens a bond based on neighboring atoms
- angular dependence helps stabilize covalent crystal structures such as diamond-cubic Si and Ge

### How we use it

For this hackathon, Tersoff is usually **not trained from scratch**. Instead, we use a published parameter file and benchmark it directly.

For Si and Ge in LAMMPS, the documentation states that `SiCGe.tersoff` can be used for:

- pure Si
- pure Ge
- binary SiGe

### Example workflow

```lammps
units metal
atom_style atomic
read_data si_bulk.data

pair_style tersoff
pair_coeff * * SiCGe.tersoff Si

fix 1 all box/relax iso 0.0
minimize 1.0e-12 1.0e-12 10000 100000

thermo_style custom step pe lx ly lz press
compute pea all pe/atom
dump 1 all custom 1 tersoff_forces.dump id type x y z fx fy fz
```

For pure Ge, map the atom type to `Ge`. For Si-Ge systems, use both element mappings.

### Strengths and limitations

- very fast
- physically interpretable
- good baseline for covalent solids
- limited flexibility outside the parameterization domain

## 5.2 ReaxFF

### How it works

ReaxFF is a reactive bond-order force field. In LAMMPS, its energy can be decomposed into many contributions such as:

- bond energy
- atom energy
- angle energy
- torsion energy
- van der Waals energy
- Coulomb energy
- charge-equilibration energy

This richer functional form makes ReaxFF more flexible than Tersoff, especially when bonds can form or break.

### Why charge equilibration matters

ReaxFF requires a charge-aware setup. In LAMMPS this means:

- use an atom style with charge
- apply charge equilibration such as `fix qeq/reaxff`

### How we use it

For the hackathon, ReaxFF is also usually used with a **published parameter file**, not trained from scratch. The crucial requirement is that the chosen `ffield.reax` must support the chemistry being studied:

- Si
- Ge
- and Si-Ge interactions if mixed systems are used

### Example workflow

```lammps
atom_style charge
read_data sige_bulk.data

pair_style reaxff NULL
pair_coeff * * ffield.reax Si Ge

fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
fix 1 all box/relax iso 0.0
minimize 1.0e-12 1.0e-12 10000 100000

thermo_style custom step pe lx ly lz press
dump 1 all custom 1 reaxff_forces.dump id type x y z q fx fy fz
```

### Strengths and limitations

- more chemically flexible than Tersoff
- supports reactive events
- more expensive than Tersoff
- strongly dependent on the chosen parameter set
- benchmarking must clearly state which parameter file was used

## 5.3 MACE

### How it works

MACE is an equivariant neural network for atomistic systems. The core idea is:

1. build neighbor graphs around atoms
2. pass messages between atoms using symmetry-aware features
3. predict atomic energy contributions
4. sum them to obtain total energy
5. obtain forces from the derivative of energy

The MACE paper emphasizes higher-order equivariant message passing, which improves expressivity without requiring deep message-passing stacks.

### Why MACE is important here

The problem statement explicitly lists MACE as the compulsory MLIP and also notes that pre-trained MLIPs should be included in benchmarks when available.

That makes MACE the best starting point for **Checkpoint 1**.

### How we use it for Checkpoint 1

Use a pre-trained MACE foundation model first, then benchmark it against ground truth for Si and Ge.

A simple ASE-based usage is:

```python
from mace.calculators import mace_mp

calc = mace_mp()
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The MACE documentation notes that in recent versions `mace_mp()` defaults to the `medium-mpa-0` model.

### How we train or fine-tune it

MACE training requires a dataset containing structures with energies and forces, typically in extended XYZ format. The documentation shows that custom data keys can be provided for energy, forces, stresses, and virials.

Typical training ingredients:

- training structures
- validation structures
- energy labels
- force labels
- cutoff radius `r_max`
- model size and irreducible representation settings

Typical command structure:

```bash
mace_run_train \
  --name="SiGe_MACE" \
  --train_file="train.xyz" \
  --valid_file="valid.xyz" \
  --energy_key="energy" \
  --forces_key="forces" \
  --r_max=5.0 \
  --batch_size=32 \
  --max_num_epochs=200 \
  --device=cuda
```

If needed, MACE can also be fine-tuned from a foundation model rather than trained entirely from scratch.

### How we use MACE in LAMMPS

The MACE documentation provides an ML-IAP workflow. First convert the trained model:

```bash
python mace/cli/create_lammps_model.py your_trained_model.model --format=mliap
```

Then use the converted model inside LAMMPS.

### Strengths and limitations

- strong accuracy on materials problems
- symmetry-aware and physically well suited for atomistic systems
- can start from pre-trained foundation models
- training needs good reference data coverage

## 5.4 DeepMD

### How it works

DeepMD decomposes the total energy into atomic contributions. For each central atom, it constructs a local-environment representation from neighboring atoms and maps this descriptor through neural networks. The DeepMD documentation explains this in terms of:

- local environment matrix
- descriptor construction
- embedding network
- fitting network

The final total energy is a sum over atoms, and forces come from energy derivatives.

### How we train it

DeepMD training needs structures with:

- coordinates
- box
- atom types
- energies
- forces
- optionally virials

The model input typically defines:

- `type_map`
- descriptor choice such as `se_e2_a`
- neighbor selection `sel`
- cutoff parameters
- fitting network
- training and validation datasets

The official training command is:

```bash
dp --pt train input.json
```

### Freeze and test workflow

After training:

```bash
dp --pt freeze -o model.pth
dp --pt test -m model.pth -s test_data
```

Compression can also be used if needed:

```bash
dp --pt compress -i model.pth -o model-compress.pth
```

### How we use it in LAMMPS

DeepMD can be run inside LAMMPS using:

```lammps
pair_style deepmd model.pth
pair_coeff * * Si Ge
```

The documentation also shows mixed model-file support such as `.pb` and `.pth`.

### Strengths and limitations

- very strong data-driven accuracy when training data are good
- efficient deployment in LAMMPS
- depends strongly on coverage and quality of the reference dataset
- less interpretable than analytical potentials

## 6. Proper Workflow for the Project

Below is a practical workflow for the full hackathon.

## 6.1 Data preparation

1. Download equilibrium Si and Ge crystal structures from Materials Project
2. Build conventional or primitive cells consistently across all models
3. Generate strained structures around equilibrium
4. Generate displaced supercells for force benchmarking
5. Collect DFT energies and forces from databases or calculations

## 6.2 Reference labels

Use:

- relaxed lattice constant from experiment or DFT
- energy per atom or relative energy from DFT
- force vectors from distorted structures

## 6.3 Classical-potential benchmarking

1. Use published Tersoff and ReaxFF parameter files
2. Relax Si and Ge cells
3. Record lattice constant, energy, and force predictions
4. Compare to reference data

## 6.4 MLIP benchmarking

1. Start with pre-trained MACE for checkpoint 1
2. Evaluate on the same structures as the classical models
3. Train or fine-tune MACE and DeepMD if needed
4. Re-run the exact same benchmark set

## 6.5 Metrics

Use at least:

- error in lattice constant
- energy MAE/RMSE
- force MAE/RMSE
- parity plots for energy and force

If possible, also include:

- stress error
- equation-of-state curves

## 7. Checkpoint 1 Scope

Checkpoint 1 requires:

1. a writeup showing understanding of the problem statement
2. explanation of MLIPs and interatomic potentials
3. an initial result comparing ground truth with **pre-trained MACE MLIPs**

So a strong checkpoint-1 submission should include:

### A. Conceptual writeup

- what molecular dynamics needs from a potential
- what interatomic potentials are
- what MLIPs are
- why Si and Ge are good benchmark systems

### B. Methodology

- structure source
- ground-truth source
- model choice
- benchmark protocol
- metrics

### C. Initial results

- predicted lattice constant of Si and Ge from pre-trained MACE
- energy comparison on strained cells
- force comparison on displaced snapshots
- one parity plot for energies
- one parity plot for force components

### D. Short discussion

- where pre-trained MACE agrees well
- where it deviates
- why pre-training may not be fully accurate for this specific benchmark

## 10. Key Technical Points to Mention in the Report

1. Tersoff and ReaxFF are usually benchmarked using published parameter files, not retrained in a short checkpoint workflow.
2. MLIPs such as MACE and DeepMD learn the energy surface from data and return forces from energy gradients.
3. Force benchmarking must use distorted structures, not only relaxed crystals.
4. Energy should be reported per atom or relative to equilibrium.
5. The same benchmark structures must be used for all models for a fair comparison.

## 11. Conclusion

This checkpoint focuses on demonstrating a correct understanding of the problem and establishing a technically sound benchmark pipeline. The central idea is to compare classical analytical potentials and ML-based potentials on the same Si and Ge structures using lattice constant, internal energy, and forces as the evaluation targets. For checkpoint 1, the most appropriate first result is a comparison between reference data and a **pre-trained MACE model**, because that is explicitly required by the problem statement and provides a strong baseline for later comparison with DeepMD, Tersoff, and ReaxFF.

## 12. Primary References

1. MACE documentation: https://mace-docs.readthedocs.io/
2. MACE paper: https://openreview.net/forum?id=YPpSngE-ZU
3. DeePMD-kit documentation: https://docs.deepmodeling.com/projects/deepmd/en/stable/
4. Deep Potential paper: https://global-sci.com/index.php/cicp/article/view/6534
5. LAMMPS Tersoff documentation: https://docs.lammps.org/pair_tersoff.html
6. LAMMPS ReaxFF documentation: https://docs.lammps.org/pair_reaxff.html
7. LAMMPS `fix box/relax`: https://docs.lammps.org/fix_box_relax.html
8. LAMMPS `minimize`: https://docs.lammps.org/minimize.html
9. LAMMPS `compute pe/atom`: https://docs.lammps.org/compute_pe_atom.html
10. Materials Project: https://materialsproject.org/
11. OQMD: https://oqmd.org/
