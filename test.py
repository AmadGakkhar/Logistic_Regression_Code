import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(0, 100, 2)

y = x1**2 + 2*x1 + 100
print (x1)
print (y)

plt.plot(x1,y)
plt.show()

for i in range(0,50,2):
	print(i)


