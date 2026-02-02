from gpaw import restart
import numpy as np
# from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
# from gpaw.directmin.exstatetools import get_occupations, excite_and_sort
from gpaw import mom
from gpaw.mom import prepare_mom_calculation
from ase import io
from ase.optimize import LBFGS
print('step0')

def get_occupations(calc):
    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)
    return f_sn


atoms, calc = restart('../tri_gs_relax/triplet-ground-state.gpw',txt='temp-output.txt')

atoms.calc.results.pop('energy')
atoms.calc.scf.converged = False
atoms.get_potential_energy()

calc.set(
        # symmetry='off',
        txt='output.txt',
        eigensolver=FDPWETDM(maxiter_inner_loop=6,
                              excited_state=True,
                              converge_unocc=False,
                              grad_tol_inner_loop=1.0e-4,
                              need_localization=False))

f_sn = excite(atoms.calc, 0, 0, spin=(1, 1))
f_sn[1][430] = 0.5
f_sn[1][431] = 0.5
prepare_mom_calculation(calc, atoms, f_sn, use_projections=True)
e_phi4 = atoms.get_potential_energy()
atoms.get_forces()

opt = LBFGS(atoms, maxstep=0.1, trajectory='es_relax.traj', logfile='es_relax.optimize')
opt.run(fmax=0.01)
# io.write('es_relax.cif', atoms)
#
calc.write('nv-triplet-excited-relaxed.gpw', mode='all')
