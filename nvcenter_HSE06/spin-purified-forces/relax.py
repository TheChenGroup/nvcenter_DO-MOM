from ase.calculators.calculator import Calculator
import gc
# from singlet_triplet import SingletTripletCalc as STC
#from singlet_triplet_write import SingletTripletCalc as STC
from ase.io import read, write
from ase.parallel import parprint
from gpaw import GPAW, PW, restart, setup_paths
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation
import numpy as np

class STC(Calculator):
    """ Singlet(S)-Triplet(T) Spin Purifier """

    implemented_properties = ['energy', 'forces']

    def __init__(self, singlet_calc, triplet_calc):

        self.singlet_calc = singlet_calc
        self.triplet_calc = triplet_calc

        Calculator.__init__(self)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # # Singlet
        self.singlet_calc.calculate(atoms=atoms, 
                                     properties=['energy','forces'],
                                     system_changes=['positions'])

        # # Triplet
        self.triplet_calc.calculate(atoms=atoms,
                                     properties=['energy','forces'],
                                     system_changes=['positions'])

        #
        energy = 2.0 * self.singlet_calc.results['energy'] -\
                       self.triplet_calc.results['energy']
        parprint('singlet ernegy', self.singlet_calc.results['energy'])
        parprint('triplet energy', self.triplet_calc.results['energy'])
        forces = 2.0 * self.singlet_calc.results['forces'] -\
                       self.triplet_calc.results['forces']

        self.results['energy'] = energy
        self.results['forces'] = forces

    def write(self):
        self.singlet_calc.write('_singlet.gpw', 'all')
        self.triplet_calc.write('_triplet.gpw', 'all')

        del self.singlet_calc
        del self.triplet_calc

        gc.collect()

        self.singlet_calc = None
        self.triplet_calc = None

        _, self.triplet_calc = restart('_triplet.gpw', txt='_triplet.txt')
        _.get_potential_energy()
        _.get_forces()

        _, self.singlet_calc = restart('_singlet.gpw', txt='_singlet.txt')
        _.get_potential_energy()
        _.get_forces()

    def write_save(self, name):
        self.singlet_calc.write(name+'_singlet.gpw', 'all')
        self.triplet_calc.write(name+'_triplet.gpw', 'all')

        del self.singlet_calc
        del self.triplet_calc

        self.singlet_calc = None
        self.triplet_calc = None

    def read_save(self, name, txt):
        _, self.singlet_calc = restart(name+'_singlet.gpw', txt=txt+'_singlet.txt')
        self.singlet_calc.calculate(_, ['forces'])

        _, self.triplet_calc = restart(name+'_triplet.gpw', txt=txt+'_triplet.txt')
        self.triplet_calc.calculate(_, ['forces'])


# Singlet calc
atoms, singlet_calc = restart('../tri_gs_relax/triplet-ground-state.gpw', txt = 'mixed.txt')

for atom in atoms:
    if atom.symbol == 'N':
        atom.magmom = 0

# singlet_calc.calculate(properties=['energy'], system_changes=['positions'])
singlet_calc = GPAW(xc='HSE06',
            mode=PW(600),
            spinpol=True,
            charge=-1,
            nbands=444,
            symmetry='off',
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            eigensolver=FDPWETDM(linesearch_algo={'name': 'max-step',
                                                  'max_step': 0.25},
                                 converge_unocc=False),
            txt='mixed.txt')


atoms.calc = singlet_calc
e_phi3_s = atoms.get_potential_energy()

for kpt in singlet_calc.wfs.kpt_u:
    if kpt.s == 0:
        s = 1
    else:
        s = -1
    homo = int(sum(kpt.f_n)) - 1
    pp = (kpt.psit_nG[homo] + s * kpt.psit_nG[homo + 1]) / np.sqrt(2)
    pm = (kpt.psit_nG[homo] - s * kpt.psit_nG[homo + 1]) / np.sqrt(2)
    kpt.psit_nG[homo][:] = pm.copy()
    kpt.psit_nG[homo + 1][:] = pp.copy()

atoms.calc.results.pop('energy')
atoms.calc.scf.converged = False

f_sn = excite(singlet_calc, 0, -1, (0, 0))
parprint('f_sn',f_sn)

singlet_calc.set(eigensolver=FDPWETDM(maxiter_inner_loop=6,
                                      excited_state=True))

_ = prepare_mom_calculation(singlet_calc, atoms, f_sn) # this modifies calc

ES = atoms.get_potential_energy()
FS = atoms.get_forces()
parprint(ES)

# Triplet calc
atoms, triplet_calc = restart('../tri_gs_relax/triplet-ground-state.gpw', txt = 'triplet.txt')
# triplet_calc.calculate(properties=['energy'], system_changes=['positions'])

atoms.calc.results.pop('energy')
atoms.calc.scf.converged = False

ET = atoms.get_potential_energy()
FT = atoms.get_forces()
parprint(ET)

# Singlet-Triplet Calc
atoms = read('../tri_gs_relax/defect_600.traj')
calc = STC(singlet_calc, triplet_calc)
atoms.calc = calc

from ase.optimize import LBFGS
dyn = LBFGS(atoms, trajectory='spin-purified.traj')
dyn.attach(atoms.calc, interval=2)

dyn.run(fmax=0.01)
