import numpy as np

from ase.parallel import paropen

from gpaw import restart, GPAW, PW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation


def update_proj(wfs):
    for kpt in wfs.kpt_u:
        wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)


def initialize_orbitals_mixedspin_singlet(wfs):
    for kpt in wfs.kpt_u:
        if kpt.s == 0:
            s = 1
        else:
            s = -1
        homo = int(sum(kpt.f_n)) - 1
        pp = (kpt.psit_nG[homo] + s * kpt.psit_nG[homo + 1]) / np.sqrt(2)
        pm = (kpt.psit_nG[homo] - s * kpt.psit_nG[homo + 1]) / np.sqrt(2)
        kpt.psit_nG[homo][:] = pm.copy()
        kpt.psit_nG[homo + 1][:] = pp.copy()


def initialize_bs_singlet(wfs):
    for kpt in wfs.kpt_u:
        if kpt.s == 0:
            psit_0 = kpt.psit_nG[:].copy()
        else:
            kpt.psit_nG[:] = psit_0
    update_proj(wfs)


def force_calc(atoms):
    atoms.calc.results.pop('energy')
    atoms.calc.scf.converged = False



atoms, calc = restart('../tri_gs_relax/triplet-ground-state.gpw')

xc = 'HSE06'

# Doubly excited singlet state symmetric
for atom in atoms:
    if atom.symbol == 'N':
        atom.magmom = 0
# A larger max step length of 0.25 seems to help 
# converging to the correct symmetric doubly excited 
# and spin-mixed solutions
calc = GPAW(xc=xc,
            mode=PW(600),
            spinpol=True,
            charge=-1,
            nbands=-12,
            symmetry='off',
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            eigensolver=FDPWETDM(linesearch_algo={'name': 'max-step',
                                                  'max_step': 0.25},
                                 converge_unocc=False),
            txt='nvcenter_es_singlet.txt')
atoms.calc = calc
e_phi3_s = atoms.get_potential_energy()

# Spin-mixed excited state mphi2 (arXiv:2303.03838v2)
initialize_orbitals_mixedspin_singlet(calc.wfs)
force_calc(atoms)
e_phi2 = atoms.get_potential_energy()
calc.write('mphi2.gpw', mode='all')


# Doubly excited singlet state broken symmetry 1phi3 (arXiv:2303.03838v2)
initialize_bs_singlet(calc.wfs)
force_calc(atoms)
e_phi3_bs = atoms.get_potential_energy()
calc.write('1phi3.gpw', mode='all')
