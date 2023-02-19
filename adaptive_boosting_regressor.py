import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(123)

x = np.arange(0, np.pi/2, 0.1).reshape(-1, 1)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

plt.plot(x, y)
plt.grid()
plt.show()

T = 9                    # the number of algorithms in the composition 
max_depth = 2            
algs = []                
s = np.array(y.ravel())  # initialize the remainders

for n in range(T):
    # create and fit algorithm
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(x, s)          

    s -= algs[-1].predict(x)    # recalculate the remainders

# restore the original signal (graph) by the set of received trees
yy = algs[0].predict(x)
for n in range(1, T):
    yy += algs[n].predict(x)

# display the results in the form of graphs
plt.plot(x, y, c='blue')      # original plot (blue)
plt.plot(x, yy, c='red')      # restored plot (red)
plt.plot(x, s, c='green')     # remainders plot (green)
plt.grid()
plt.show()

