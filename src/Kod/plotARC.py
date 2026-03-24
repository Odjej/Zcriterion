
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/loe/Desktop/DREAM/py/')

from DREAM.DREAMOutput import DREAMOutput
print("hejhej")
plt.rcParams.update({'font.size': 14})

do = DREAMOutput('output.h5')

#fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

print(do.grid.t)
# Runaway rate
I_re = do.eqsys.j_re.current()
Temp = do.eqsys.T_cold.data
plt.plot(Temp[:], I_re, linewidth=2, color='k')
#plt.xlim([0,do.grid.t[-1]])
#plt.ylim([0, 2])
plt.xlabel('Temperatur (eV)')
plt.ylabel('Runaway ström (A)')
plt.title('Runaway ström')

plt.tight_layout()
plt.show()
