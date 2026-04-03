# Hackathon Summary and Methodology

## Assumption

In your message you wrote `MASE`, but the problem statement and resource list clearly mention `MACE` as the compulsory MLIP. In this note I therefore treat `MASE` as `MACE`.

## Problem Summary

The problem is about benchmarking atomistic models for **Si** and **Ge**. The team has to compare:

1. Classical interatomic potentials such as **Tersoff** and **ReaxFF**
2. Machine-learning interatomic potentials (MLIPs) such as **MACE** and **DeepMD**

against **ground-truth data** from experiment and/or DFT databases.

The target properties are:

1. **Lattice constant**
2. **Internal energy**
3. **Atomic forces**

The goal is not only to generate numbers, but to justify which model class gives the best agreement and why.

## Short Explanation of the Physics

Molecular dynamics needs a function that maps atomic positions to total energy. Once energy is known, forces follow from:

\[
\mathbf{F}_i = - \frac{\partial E}{\partial \mathbf{R}_i}
\]

So all models in this hackathon are really trying to approximate the same physical object: the **potential energy surface**.

- **Lattice constant** tells us the equilibrium size of the crystal.
- **Internal energy** tells us how stable a structure is.
- **Forces** tell us how atoms want to move when the structure is distorted.

If a model predicts energy correctly but not forces, it is not reliable for dynamics. If it predicts forces well but gives the wrong lattice constant, it does not reproduce the correct equilibrium structure. That is why all three properties matter.

## What Interatomic Potentials and MLIPs Are

### Interatomic potentials

These are analytical or semi-empirical equations with fitted parameters.

- **Tersoff** is a bond-order potential. It combines repulsive and attractive pair terms with an environment-dependent bond-order term, so the strength of a bond changes depending on nearby atoms and bond angles.
- **ReaxFF** is a reactive force field. It uses bond-order ideas too, but adds many more contributions such as bond, angle, torsion, van der Waals, Coulomb, and charge-equilibration terms. It can model bond breaking and formation.

### MLIPs

These learn the energy surface directly from reference data.

- **MACE** is an equivariant graph neural network. It learns local atomic environments and predicts total energy while respecting rotational symmetry.
- **DeepMD** maps each local atomic environment into symmetry-preserving descriptors and feeds them into neural networks to predict atomic contributions to the total energy.

## Recommended Methodology

### 1. Define the reference structures

Use the equilibrium diamond-cubic structures of Si and Ge as the starting point.

### 2. Build a benchmark dataset

For a meaningful benchmark, do not test only the perfect equilibrium cell.

- For **lattice constant**:
  relax the unit cell and compare the predicted equilibrium lattice parameter with reference values.
- For **energy**:
  generate isotropically strained cells around equilibrium, for example from `0.94 a0` to `1.06 a0`, and compare energy per atom.
- For **forces**:
  generate off-equilibrium structures by adding small random displacements to atoms in a supercell. This is important because forces in the perfectly relaxed diamond structure should be nearly zero by symmetry.

### 3. Obtain reference data

Use experimental and/or DFT-backed databases.

- **Materials Project** for relaxed structures and energies
- **OQMD** for additional DFT structure-energy data
- For **forces**, prefer datasets or calculations that contain force labels for distorted snapshots. Static relaxed crystal entries alone are not enough for a real force benchmark.

### 4. Evaluate each model

For every method, compute:

1. Relaxed lattice constant
2. Energy per atom or relative energy
3. Forces on distorted structures

### 5. Compare with clear metrics

Use:

- **Absolute or relative error** for lattice constant
- **MAE/RMSE** for energy per atom
- **MAE/RMSE** for force components
- **Parity plots** for energy and force

For lattice constant, a table or bar chart is usually clearer than a parity plot because there are only a few values.

## Property Definitions and How to Compute Them

### Lattice constant

For diamond-cubic Si and Ge, the conventional cubic cell is described by one lattice parameter `a`.

Two clean ways to obtain it:

1. Relax the box and atoms until the stress is near zero
2. Compute an energy-volume curve and fit the minimum

In LAMMPS, the simplest route is usually:

1. read the structure
2. apply the model
3. run `minimize`
4. use `fix box/relax` for variable-cell relaxation
5. read `lx = ly = lz` for the relaxed cubic cell

### Internal energy

For this benchmark, the safest reporting choice is:

- **Potential energy per atom** at equilibrium, and/or
- **Relative energy** with respect to the equilibrium structure

For same-composition comparisons, relative energy curves are often more robust than raw absolute energies.

### Forces

Forces are the negative energy gradients with respect to atomic positions. They are the most direct test of whether a model reproduces local atomic interactions.

Important point:

- If you test only the fully relaxed bulk structure, the reference forces will be approximately zero.
- To test force quality properly, you need **distorted snapshots**.

## How Each Method Should Be Used in This Project

### Tersoff

- Use a published parameter file that supports Si and Ge.
- In LAMMPS, `SiCGe.tersoff` is an official file that can be used for pure Si, pure Ge, and Si-Ge according to the LAMMPS documentation.
- Best for fast baseline benchmarking.

### ReaxFF

- Use a validated `ffield.reax` parameter set that contains Si and Ge chemistry.
- ReaxFF requires charge handling and charge equilibration in the simulation setup.
- It is more expensive than Tersoff and much more parameter-sensitive.

### MACE

- For checkpoint 1, start with the official **pre-trained foundation model**.
- Use it first without re-training, because the problem statement explicitly asks for an initial comparison with pre-trained MACE MLIPs.
- Later, if needed, fine-tune on Si/Ge-specific DFT data.

### DeepMD

- Prepare a training dataset with structures, energies, forces, and optionally virials/stresses.
- Train on local atomic environments.
- Freeze the trained model and run it in LAMMPS for unified comparison.

## Recommended Workflow for Checkpoint 1

Checkpoint 1 does **not** need the full final benchmark. It mainly needs:

1. Understanding of the problem statement
2. Explanation of MLIPs and interatomic potentials
3. Initial results comparing reference data with a **pre-trained MACE model**

So the best checkpoint-1 workflow is:

1. Write the conceptual explanation
2. Build Si and Ge reference structures
3. Generate strained and displaced snapshots
4. Evaluate a pre-trained MACE model
5. Compare against ground truth
6. Add one or two parity plots and a short interpretation

## Best Practices for Your Report

- Clearly distinguish **reference data** from **model predictions**
- Use **energy per atom**, not only total energy
- Mention that force benchmarking needs **off-equilibrium** structures
- State whether a model is **pre-trained**, **fine-tuned**, or only used with a published parameter file
- Mention that Tersoff and ReaxFF are normally **used with published parameters**, not trained from scratch in a short hackathon workflow

## One-Line Conclusion

The core idea of the project is to benchmark how well classical analytical potentials and data-driven MLIPs reproduce the same DFT/experimental potential energy surface of bulk Si and Ge, using lattice constant, energy, and forces as the comparison targets.
