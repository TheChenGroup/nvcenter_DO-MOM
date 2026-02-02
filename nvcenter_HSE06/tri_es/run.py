import numpy as np

from ase.parallel import paropen

from gpaw import restart, GPAW, PW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation

atoms, calc = restart('../tri_gs_relax/triplet-ground-state.gpw', txt='output.txt')

atoms.calc.results.pop('energy')
atoms.calc.scf.converged = False
atoms.get_potential_energy()

calc.set(txt='diamond_es_triplet.txt')
calc.set(eigensolver=FDPWETDM(maxiter_inner_loop=6,
                              excited_state=True,
                              converge_unocc=False,
                              grad_tol_inner_loop=1.0e-4,
                              need_localization=False))
f_sn = excite(atoms.calc, 0, 0, spin=(1, 1))
prepare_mom_calculation(calc, atoms, f_sn, use_projections=True)
e_phi4 = atoms.get_potential_energy()


calc.write('nv-triplet-excited.gpw', mode='all')

