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
from DREAM.Output.ScalarQuantity import ScalarQuantity

ds = DREAMSettings () 

#Data=np.array(['Temp','D0', 'D0_jon', 'D', 'D_jon', 'T', 'T_jon', 'Ne', 'Ne_jon1', 'Ne_jon2', 'Ne_jon3', 'Ne_jon4', 'Ne_jon5', 'Ne_jon6', 'Ne_jon7', 'Ne_jon8', 'Ne_jon9', 'Ne_jon10', 'z', 'n_D', 'n_Ne'])

#for s in Data:
#    globals()[s]=np.load(f'Temp och Z/{s}.npy')

#När ny simulering körs behöver denna ändras på, vilken map output sparas i, och vad filen heter
data = np.load(r'/home/loe/Desktop/DREAM/examples/ARC/Temp och Z/Temp och Z_100.npz')

globals().update(data)
#laddningstätheterna, vänster index är neon, höger är deuterium
n_D0_joner = np.array([D0, D0_jon]) 
n_D_joner=np.array([D,D_jon])
n_T_joner = np.array([T, T_jon])
n_Ne_joner= np.array([Ne, Ne_jon1, Ne_jon2, Ne_jon3, Ne_jon4, Ne_jon5, Ne_jon6, Ne_jon7, Ne_jon8, Ne_jon9, Ne_jon10])

#print(D)

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




#svep
runaways = np.zeros_like(D)
Ips = np.zeros_like(D)
Ip0 = np.zeros_like(D)
Dreicer_RE = np.zeros_like(D)
Compton_RE = np.zeros_like(D)
Tritium_RE = np.zeros_like(D)
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        ds = DREAMSettings () 
        
        # E-field 
        ds.eqsys.E_field.setType(Efield.TYPE_SELFCONSISTENT)
        ds.eqsys.E_field.setBoundaryCondition(Efield.BC_TYPE_SELFCONSISTENT, inverse_wall_time=0)
        ds.eqsys.j_ohm.setInitialProfile(j0, radius=r, Ip0=Ip)
        
            # Set temperature
        ds.eqsys.T_cold.setPrescribedData(temperature=Temp[i][j])  #vänster index är neon, höger är deuterium
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
        
        
        
        #print(np.sum(n_D_joner[:,i,j]))
        #n_T_joner[:, 0, 0].reshape((2,1,1))
        #Add ion species to list of ions
        ds.eqsys.n_i.addIon('T', Z=1, iontype = Ions.IONS_PRESCRIBED, n=n_T_joner[:, i, j].reshape((2,1,1)), 
        r = np.array([0]), t = np.array([0]), tritium=True, opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE) 
        ds.eqsys.n_i.addIon('D0', Z=1, iontype = Ions.IONS_PRESCRIBED, n=n_D0_joner[:, i, j].reshape((2,1,1)),
        r = np.array([0]), t = np.array([0]), opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        ds.eqsys.n_i.addIon('D', Z=1, iontype = Ions.IONS_PRESCRIBED, n=n_D_joner[:, i, j].reshape((2,1,1)),
        r = np.array([0]), t = np.array([0]), opacity_mode=Ions.ION_OPACITY_MODE_GROUND_STATE_OPAQUE)
        ds.eqsys.n_i.addIon(name='Ne', Z=10, iontype=Ions.IONS_PRESCRIBED, n=n_Ne_joner[:, i, j].reshape((11,1,1)), 
        r = np.array([0]), t = np.array([0]))
        
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
        ds.timestep.setNumberOfSaveSteps(1000) #hur mågna steps vi sparar, mindre än nt för att spara plats
                # Save settings to HDF5 file
        ds.save(f'/home/loe/Desktop/DREAM/examples/ARC/Settings/settings{i,j}.h5')

        do = runiface(ds, f"/home/loe/Desktop/DREAM/examples/ARC/Output100/output_{i,j}.h5")
        
        # spara I_re och I_p
        I_re = do.eqsys.j_re.current()
        time_grid = do.grid  
        data_array = do.eqsys.I_p.getData()  

        #Ip = do.eqsys.I_p.data
        #Ip = ScalarQuantity(name='I_p', data=data_array, grid=time_grid, output=do)
        #print(Ip)
        #print(Ip[0].item())
        print(np.max(I_re))
        runaways[i,j] = np.max(I_re)
        max_index = np.argmax(I_re)
        print(max_index)
        
        Dreicer_RE[i,j] = np.sum(do.other.fluid.gammaDreicer[:,0] * np.diff(do.grid.t[:]))
        Compton_RE[i,j] = np.sum(do.other.fluid.gammaCompton[:,0] * np.diff(do.grid.t[:]))
        Tritium_RE[i,j] = np.sum(do.other.fluid.gammaTritium[:,0] * np.diff(do.grid.t[:]))
        Ip0[i,j] = data_array[0].item()
        Ips[i,j] = data_array[max_index].item()
        

#print(runaways)

np.savez(r'/home/loe/Desktop/DREAM/examples/ARC/Data/runaways100.npz', runaways = runaways, Ip0 = Ip0, Ips = Ips, n_D = n_D, n_Ne = n_Ne, Dreicer_RE = Dreicer_RE, Compton_RE = Compton_RE, Tritium_RE  = Tritium_RE)
print('Klar')

