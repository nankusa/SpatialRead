import math
import numpy as np
from ase.build import make_supercell
from ase.data import chemical_symbols


def _make_supercell(atoms, cutoff=None, scale=None):
    """
    make atoms into supercell when cell length is less than cufoff (min_length)
    """
    assert cutoff is not None or scale is not None, f"One of cutoff and scale must not be None"
    if scale is not None:
        m = np.zeros([3, 3])
        np.fill_diagonal(m, scale)
        atoms = make_supercell(atoms, m)
        return atoms

    # when the cell lengths are smaller than radius, make supercell to be longer than the radius
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l < cutoff:
            scale_abc.append(math.ceil(cutoff / l))
        else:
            scale_abc.append(1)

    # make supercell
    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms

def atomic_number_to_symbol(atomic_number):
    return chemical_symbols[atomic_number]
