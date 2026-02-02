from gpaw import GPAW, PW, restart
import datetime
from ase import io
from ase.optimize import LBFGS

molname = 'diamond'

ecut=600

atoms, calc = restart('./pbe-gs-relax/nvcenter_gs_relax.gpw',txt='output.txt')

atoms.calc.results.pop('energy')
atoms.calc.scf.converged = False
calc.set(xc='HSE06')

atoms.get_potential_energy()
atoms.get_forces()

#
opt = LBFGS(atoms, trajectory='defect_{}.traj'.format(ecut), logfile='defect_{}.optimize'.format(ecut))
opt.run(fmax=0.01)
io.write('relax-defect_{}.cif'.format(ecut), atoms)
calc.write('triplet-ground-state.gpw', mode='all')


