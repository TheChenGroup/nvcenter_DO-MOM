from gpaw import GPAW, PW
from ase import io
from ase.optimize import LBFGS
from gpaw.directmin.etdm_fdpw import FDPWETDM
"""Optimizes the geometry of the triplet ground state of the NV center 
   defect in diamond using direct minimization with PW basis and the r2SCAN 
   functional (see arXiv:2303.03838v2)

   Input: 
       diamond.traj 
   Trajectory of geometry optimization of a small diamond cell

   Output:
       nvcenter_gs_relax.gpw
   gpw file at the optimized geometry read by the nvcenter_exc.py script
   that computes the excitation energies of the NV center
"""

# xc = 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
# xc = 'HSE06'
xc = 'PBE'

name = 'diamond'
atoms = io.read('my_hse06_diamond_216_vasp.traj')
outname = 'nvcenter_gs_relax'

atoms = atoms.repeat((3, 3, 3))
#io.write('hse_nvcenter_POSCAR',atoms)
atoms[213].symbol = 'N'
atoms[213].magmom = 2.0
atoms.pop(208)

io.write('hse_nvcenter_POSCAR',atoms)

#print('start calc')
calc = GPAW(xc=xc,
            mode=PW(600),
            spinpol=True,
            charge=-1,
            nbands=-12,
            symmetry='off',
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            eigensolver=FDPWETDM(maxiter_inner_loop=6,
                              excited_state=False,
                              converge_unocc=True,
                              grad_tol_inner_loop=5.0e-4,
                              need_localization=False),
            txt=outname+'.txt',
            )
atoms.calc = calc
atoms.get_potential_energy()
atoms.get_forces()

#print('start opt')
opt = LBFGS(atoms, trajectory=outname+'.traj', logfile=outname+'.log')
opt.run(fmax=0.01)
calc.write(outname+'.gpw', mode='all')

#print('finish opt')
