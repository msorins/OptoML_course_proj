# OptoML_course_proj

This project is done as part of the Optimization for Machine Learning (CS-439) course at EPFL in the summer 2021 semester. 
The goal of the project was to compare how zero-order optimization methods perform in comparison to standard first-order methods on classification and regression tasks.

## Using the code
The code and data for the classification and regression tasks are contained in folders with the according names. 

#### Classification:
The zero-order method analysed for this task is random search with gradient oracle as seen in ["Random Gradient-Free Minimization of Convex Functions" by
Y. Nesterov and V. Spokoiny](https://link.springer.com/article/10.1007/s10208-015-9296-2). The first-order method used for comparison is standard SGD.
In order to run all necessary experiments and generate plots, run the `classification.ipynb` notebook.
This will prepare the data, run hyperparameter tuning and training with the tuned parameter, and generate all necessary plots.

#### Regression:
The zero-order method analysed for regression is Particle Swarm Optimization (PSO), while the first-order optimizer used is ADAM. 
The regression experiments are divided into two notebooks - `regression.ipynb`, which runs training and generates plots for the first-order method,
and `regression_pso.ipynb`, which does the same for the zero-order method. In both cases, the data is preprocessed in the notebook before any experiments are run.  

## Necessary libraries:
The necessary libraries for running this code are stated in the `requirements.txt` file. All necessary imports are done at the beginninf=g of each notebook or .py file.
