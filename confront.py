from matplotlib import pyplot as plt
import numpy as np



h=np.loadtxt('h.txt')
nh=np.loadtxt("nh.txt")
ks=[.1,.3,.5,.7,.9,1,1.1,1.3,1.5,2,3,4,5,6,7]

plt.subplot() 
plt.plot(ks,nh[0],label="NH Single round",color="C2") 
plt.plot(ks,h[0],label="H  Single round",color="C2",linestyle="-.")  
plt.plot(ks,nh[1],label="NH Multiple round",color="C3") 
plt.plot(ks,h[1],label="H  Multiple round",color="C3",linestyle="-.") 
plt.plot(ks,nh[2],color="C0",label="NH total")
plt.plot(ks,h[2],color="C0",label="H  total",linestyle="-.") 
plt.xlim((0,3))
plt.ylim((0,2))
 


plt.axhline(y=1/.7,linestyle=":",c="purple",label="estimated performance",xmax=1)
plt.title("throughput homogeneous VS non-homogeneous")
plt.ylabel("throughput")
plt.xlabel("K")
plt.legend()
plt.show()