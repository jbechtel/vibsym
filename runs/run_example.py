from vibsym.plots import plot_normal_modes
from vibsym.example_reps import get_example_molecule, ExampleMoleculeType

mol_type = ExampleMoleculeType.SQUARE
mol = get_example_molecule(mol_type)
Q, dims = mol.find_normal_modes()
plot_normal_modes(mol.coords, Q, dims)

