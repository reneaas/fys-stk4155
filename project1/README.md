## Code documentation

The important codes for this project is found [here](https://github.com/reneaas/fys-stk4155/tree/master/project1/codes).
### Structure of the codes

We've developed a set of classes with the following structure:

1. A superclass called [Regression](https://github.com/reneaas/fys-stk4155/blob/master/project1/codes/regression.py).
    - This class contains class methods that are used by all algorithms.
2. A derived class called [OLS](https://github.com/reneaas/fys-stk4155/blob/master/project1/codes/ols.py)
    - This derived class implements the *ordinary least squares* method.
    - It inherits all methods declared in *regression.py*.
3. Another derived class called [Ridge](https://github.com/reneaas/fys-stk4155/blob/master/project1/codes/ridge.py)
    - Implements Ridge regression.
    - Inherits all methods declared in *regression.py*
4. A derived class called [Lasso](https://github.com/reneaas/fys-stk4155/blob/master/project1/codes/lasso.py)
    - Implements Lasso regression using the Scikit-Learn module.
    - Inherts all methods declared in *regression.py*

### Producing data

1. To produce data for the Franke function, run the script *generate_data.py*, with the following in a Linux/Unix command line:

```terminal
python3 generate_data.py N sigma
```
  - *N* is the number of points
  - *sigma* is the desired standard deviation.

2. Extracting the image used in the [report](https://github.com/reneaas/fys-stk4155/blob/master/project1/report/Project_1___Linear_Regression.pdf) from the terrain data is done using the script *terrain.py*. Simply run the following in a Linux/Unix command line:

```terminal
python3 terrain.py
```

  - This produces an image of 1000 x 1000 pixels.


### Usage of codes

The usage of the derived classes are identical, and they all have access to the methods in the Regression superclass.

#### Standard training and testing of the model:

Given a datafile with tuples (x, y, f(x, y)) on each line, the usage of each class is pretty straight forward.

1. OLS
  ```Python
  from ols import OLS
  my_solver = OLS() #Initiates the solver.
  my_solver.read_data(filename) #Reads data and scales it according to Z-score
  my_solver.create_design_matrix(deg) #Creates design matrix for a polynomial of degree deg
  my_solver.split_data()
  my_solver.train()  #Computes the parameters of the model
  R2, MSE = my_solver.predict_test() #Computes R2-score and MSE on the test data.
  ```

2. Ridge
```Python
from ridge import Ridge
my_solver = Ridge(Lambda = value) #Initiates the solver with a given value for the regularization parameter.
my_solver.read_data(filename) #Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg) #Creates design matrix for a polynomial of degree deg
my_solver.split_data()
my_solver.train()  #Computes the parameters of the model
R2, MSE = my_solver.predict_test() #Computes R2-score and MSE on the test data.
```
  - If the value of Lambda is not initialized, it's set to *Lambda = None* by default. You can then specify it at any point after initialization with

    ```Python
    my_solver.Lambda = value
    ```
  - Note that *train* and *predict_test* will not work if Lambda is not specified.
3. Lasso
```Python
from lasso import Lasso
my_solver = Lasso(Lambda = value) #Initiates the solver with a given value for the regularization parameter.
my_solver.read_data(filename) #Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg) #Creates design matrix for a polynomial of degree deg
my_solver.split_data()
my_solver.train()  #Computes the parameters of the model
R2, MSE = my_solver.predict_test() #Computes R2-score and MSE on the test data.
```

  - Here too, *Lambda = None* by default. You can specify Lambda at any point after initiation of the solver by
    ```Python
    my_solver.Lambda = value
    ```
  - Note that *train* and *predict_test* will not work if Lambda is not specified.


#### Bootstrap analysis
If the sample size of the dataset is limited, bootstrap analysis can be perfomed using *any* of the solvers, initiated as shown above, by the following code segment

```Python
R2, MSE, bias, variance = my_solver.bootstrap(B)
```

  - *B* is the number of bootstrap samples.
  - The methods returns an average R2-score, MSE, bias and variance computed on the test data.

#### k-fold cross-validation
To perform k-fold cross-validation on the whole dataset, the initialization of the solvers as above combined with the following code-segment gives you the computed performance metrics R2 and MSE:

```Python
R2, MSE = my_solver.k_fold_cross_validation(k)
```
  - *k* is the number of folds to perform crossvalidation for.
  - It returns the average R2-score and MSE on thetest data.
  - The k-fold cross-validation is performed on the full data set.

### Test program

As as simple test program, we've implemented [test_program.py](https://github.com/reneaas/fys-stk4155/blob/master/project1/codes/test_program.py) which show-cases all of the above using a generated dataset on the Franke function. Simply run the follow command in any Linux/Unix command line

```terminal
python3 test_program.py
```

which should give the output similar to

```terminal
OLS; R2 score =  0.8841606202925218
OLS; MSE =  0.11085594125736685
Ridge; R2 score =  0.913052223133676
Ridge; MSE =  0.0967329921906487
Lasso; R2 score =  0.8829292527756718
Lasso; MSE =  0.12034045376909479
Bootstrap Ridge; R2 =  0.8748054610717052
Bootstrap Ridge; MSE =  0.11434125675837999
Bootstrap Ridge; Bias =  0.1109677317344068
Bootstrap Ridge; Variance =  0.003373525023973192
10-fold cross-validation Ridge; R2 =  0.8226413708891136
10-fold cross-validation Ridge; MSE =  0.1197305186669059,
```
