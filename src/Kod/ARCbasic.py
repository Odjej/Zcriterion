import numpy as np
import sys
sys.path.append('/home/loe/Desktop/DREAM/py/')

import matplotlib.pyplot as plt
from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ElectricField as Efield
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Equations.IonSpecies as Ions

ds = DREAMSettings () 

Data=np.array(['Temp','D', 'D_jon', 'T', 'T_jon', 'Ne', 'Ne_jon1', 'Ne_jon2', 'Ne_jon3', 'Ne_jon4', 'Ne_jon5', 'Ne_jon6', 'Ne_jon7', 'Ne_jon8', 'Ne_jon9', 'Ne_jon10', 'z', 'n_D', 'n_Ne'])

for s in Data:
    globals()[s]=np.load(f'Temp och Z/{s}.npy')

n_D_joner = np.array([D, D_jon])
n_T_joner = np.array([T, T_jon])
n_Ne_joner= np.array([Ne, Ne_jon1, Ne_jon2, Ne_jon3, Ne_jon4, Ne_jon5, Ne_jon6, Ne_jon7, Ne_jon8, Ne_jon9, Ne_jon10])



#ARC parametrar
a  = 1.18  # minor radius (m)
r  = np.linspace(0,a)
b  = 1.1215 * a  # minor radius of tokamak wall (m), a_wall från Björn
B0 = 11.4     # toroidal magnetic field on-axis (T)
R0 = 4.62    # major radius (m)
Ip = 12e6   # Target plasma current (A)
j0 = Ip / (np.pi * a**2)  # current density (A/m^2) 

# Simulation parameters
ne0 = 3e20 # electron density (m^-3)
tmax = 5e-2
nt = 10000

n = 0
d = 2
i = (slice(None), n, d)
j = (d,n)


 # E-field 
ds.eqsys.E_field.setType(Efield.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setBoundaryCondition(Efield.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0)
ds.eqsys.j_ohm.setInitialProfile(j0, radius=r, Ip0=Ip)

        # Set temperature
ds.eqsys.T_cold.setPrescribedData(temperature=Temp[j]) 
        #räkna ut med equilibriumtemperaturen

        #Comptonparamterar
Phi0 = 1.5e16
c1 = 1.922
c2 = 0.954
c3 = 0.041

        #seed
ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK) #undersök den andra också
ds.eqsys.n_re.setTritium(tritium=Runaways.TRITIUM_MODE_FLUID) 
ds.eqsys.n_re.setCompton(compton=Runaways.COMPTON_MODE_FLUID, photonFlux = Phi0,
C1=c1, C2 = c2, C3 = c3)
ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
ds.eqsys.n_re.setHottail(Runaways.HOTTAIL_MODE_DISABLED)
#n_T_joner[:, 0, 0].reshape((2,1,1))
#Add ion species to list of ions
ds.eqsys.n_i.addIon('T', Z=1, iontype = Ions.IONS_PRESCRIBED, n=n_T_joner[i].reshape((2,1,1)), r = np.array([0]), t = np.array([0]), tritium=True) 
ds.eqsys.n_i.addIon('D', Z=1, iontype = Ions.IONS_PRESCRIBED, n=n_D_joner[i].reshape((2,1,1)),r = np.array([0]), t = np.array([0]))
ds.eqsys.n_i.addIon(name='Ne', Z=10, iontype=Ions.IONS_PRESCRIBED, n=n_Ne_joner[i].reshape((11,1,1)), r = np.array([0]), t = np.array([0]))

        # Hot-tail grid settings
ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER ####
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED ####
ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT ####
ds.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONAL ####

ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

        # Set up radial grid 
ds.radialgrid.setB0(B0)
ds.radialgrid.setMinorRadius(a)
ds.radialgrid.setWallRadius(b)
ds.radialgrid.setNr(1) #pga 0d

        # Use the linear solver
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.other.include('fluid') #other.fluid.gammaDreicer.plot() för att kolla dreicergenereringen tex

        # Set time stepper
ds.timestep.setTmax(tmax) #fundera på tmax
ds.timestep.setNt(nt) #fundera på nt

        # Save settings to HDF5 file
ds.save('settings.h5')

do = runiface(ds, 'output.h5')
#hitta maxvärdet för I_re och lägg till i en array
    #out = DREAMOutput(outfile)
max_I_re = []
I_re = do.eqsys.j_re.current()

print(Temp[j])
print(n_T_joner[i])
print(n_T_joner)
