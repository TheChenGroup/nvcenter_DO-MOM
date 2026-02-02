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
#print('step0')

def get_occupations(calc):
    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)
    return f_sn

atoms, calc = restart(
        '../tri_es/nv-triplet-excited.gpw', 
        txt='output.txt', 
        eigensolver=FDPWETDM(maxiter_inner_loop=6,
                              excited_state=True,
                              converge_unocc=True,
                              grad_tol_inner_loop=1.0e-4,
                              need_localization=False)
        )
atoms.get_potential_energy()
atoms.get_forces()

#print('step1')
f_sn = get_occupations(calc)
prepare_mom_calculation(calc, atoms, f_sn, use_projections=True)

#print('step2')
calc.set(
        eigensolver=FDPWETDM(maxiter_inner_loop=6,
                              excited_state=True,
                              converge_unocc=False,
                              grad_tol_inner_loop=1.0e-4,
                              need_localization=False)
)

atoms.get_potential_energy()
atoms.get_forces()

# atoms_tmp = io.read('../../../LDA/triplet_x/relax_geometry_tr/defect_excited_triplet.traj')
# atoms.set_positions(atoms_tmp.get_positions())
# atoms.get_potential_energy()
# atoms.get_forces()
#
ecut = 'excited_triplet'
opt = LBFGS(atoms, maxstep=0.1, trajectory='defect_{}.traj'.format(ecut), logfile='defect_{}.optimize'.format(ecut))
opt.run(fmax=0.01)
io.write('relax-defect_{}.cif'.format(ecut), atoms)
#
calc.write('nv-triplet-excited-relaxed.gpw', mode='all')
