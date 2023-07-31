import numpy as np  
from statistics import stdev
import matplotlib.pyplot as plt

data = np.loadtxt('populatii.txt')

an, iepure, linx, morcov = data.T  # salveaza coloanele ca si variabile
print(data)
print("Punctul a)")
medie_iepuri=np.average(iepure)
print("Media iepurilor in perioada analizata este: ", round((medie_iepuri),2))

medie_linx=np.average(linx)
print("Media lincsilor in perioada analizata este: ", round((medie_linx),2))


medie_morcovi=np.average(morcov)
print("Media morcovilor in perioada analizata este: ", round((medie_morcovi),2))
print("-----")

deviatie_iepuri=stdev(iepure)
print("Variatia standard a iepurilor este: ", round((deviatie_iepuri),2))
deviatie_lincsi=stdev(linx)
print("Variatia standard a lincsilor este: ", round((deviatie_lincsi),2))
deviatie_morcovi=stdev(morcov)
print("Variatia standard a morcovilor este: ", round((deviatie_morcovi),2))
print("------")
print("Punctul b)")

max_iepuri=np.max(iepure)
max_lincsi=np.max(linx)
max_morcovi=np.max(morcov)
if max_lincsi>max_iepuri:
  max=max_lincsi
  specie="linx"
else:
  if max_morcovi>max_lincsi:
    max=max_morcovi
    specie="morcov"
  else:
    max=max_iepuri
    specie="iepure"
print("Cea mai mare populatie numara: ", max, " exemplare si apartine speciei: ", specie)
print("------")
print("Punctul c)")
pop_iepuri_peste_prag=np.any(iepure, out=print(iepure>50000), where= iepure>50000)
print(pop_iepuri_peste_prag)
print("------")
print("Punctul d)")
min_iepure=np.argsort(iepure)
min_linx=np.argsort(linx)
min_morcov=np.argsort(morcov)
print("Primii 2 ani in care populatia de iepuri a fost minima au fost: %d si %d" %(an[min_iepure[0]], an[min_iepure[1]]))
print("Primii 2 ani in care populatia de lincsi a fost minima au fost: %d si %d" %(an[min_linx[0]], an[min_linx[1]]))
print("Primii 2 ani in care populatia de morcovi a fost minima au fost: %d si %d" %(an[min_morcov[0]], an[min_morcov[1]]))
print("------")
print("Punctul e)")
plt.axes([0.2, 0.1, 0.5, 0.8])
plt.plot(an, iepure, an, linx)
plt.legend(('Iepure', 'Linx'), loc=(1.05, 0.5))
plt.show()
print()
print("Corelatia este: ")
print(np.corrcoef(iepure, linx))