# Forward Euler Scheme

To run the a test code for the forward Euler scheme, simply write

```terminal
make all
```

in the current directory. A figure of relative error should appear shortly after.

### General structure

The forward Euler algorithm is found in [functions.cpp](https://github.com/reneaas/fys-stk4155/blob/master/project3/codes/forward_euler/functions.cpp). The main code, which solves the 1D diffusion equation is [main.cpp](https://github.com/reneaas/fys-stk4155/blob/master/project3/codes/forward_euler/main.cpp), which takes three command line arguments, in the following order:

| Args    |     |
| :------------- | :------------- |
| outfilename   |  the name of the file, with proper file extension, the program writes the numerical solution for every value of x and t. |
| total_time| The total simulation time |
| dx| The step size in coordinate space |



To simply run the forward Euler scheme, and produce new results, the following comand can be run in a Linux terminal
```terminal
make
```

The python file [plot.py](https://github.com/reneaas/fys-stk4155/blob/master/project3/codes/forward_euler/plot.py) plots the relative error between the numerical and analytical solution. It requires a .txt file with results to exists, i. e. it does not produce results on its own.
