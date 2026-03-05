# Fundamentals  
This introductory module is focused on introducing several core software engineering methods for testing and debugging, and also includes some basic mathematical foundations.  
Before starting this assignment, make sure to set up your workspace following the setup guide, to understand how the code should be organized.  
  
## Guides  
Each module has a set of guides to help with the background material. We recommend working through the assignment and utilizing the guides suggested for each task.  
* Contributing  
* Functional Python  
* Property Testing  
* Modules  
* Visualization  
In addition to completing each of the tasks below, you need to ensure that all the unit tests and style checks pass in the module. This may require writing docstrings or adding types to functions.  
  
## Task 0.1: Operators  
This task is designed to help you get comfortable with style checking and testing. We ask you to implement a series of basic mathematical functions. These functions are simple, but they form the basis of MiniTorch. Make sure that you understand each of them as some terminologies might be new.  
## Todo  
Complete the following functions in minitorch/operators.py and pass tests marked as task0_1.  

| Function  | Description                                              |
| --------- | -------------------------------------------------------- |
| mul       | Multiplies two numbers                                   |
| id        | Returns the input unchanged                              |
| add       | Adds two numbers                                         |
| neg       | Negates a number                                         |
| lt        | Checks if one number is less than another                |
| eq        | Checks if two numbers are equal                          |
| max       | Returns the larger of two numbers                        |
| is_close  | Checks if two numbers are close in value                 |
| sigmoid   | Calculates the sigmoid function                          |
| relu      | Applies the ReLU activation function                     |
| log       | Calculates the natural logarithm                         |
| exp       | Calculates the exponential function                      |
| inv       | Calculates the reciprocal                                |
| log_back  | Computes the derivative of log times a second arg        |
| inv_back  | Computes the derivative of reciprocal times a second arg |
| relu_back | Computes the derivative of ReLU times a second arg       |
  
## Task 0.2: Testing and Debugging  
We ask you to implement property tests for your operators from Task 0.1. These tests should ensure that your functions not only work but also obey high-level mathematical properties for any input. Note that you need to change arguments for those test functions.  
## Todo  
Complete the test functions in tests/test_operators.py marked as task0_2.  
  
## Task 0.3: Functional Python  
To practice the use of higher-order functions in Python, implement three basic functional concepts. Use them in combination with operators described in Task 0.1 to build up more complex mathematical operations that work on lists instead of single values.  
## Todo  
Complete the following functions in minitorch/operators.py and pass tests marked as task0_3.  

| Function | Description |
| -------- | --------------------------------------------------------------------------------------- |
| map | Higher-order function that applies a given function to each element of an iterable |
| zipWith | Higher-order function that combines elements from two iterables using a given function |
| reduce | Higher-order function that reduces an iterable to a single value using a given function |
  
Using the above functions, implement:  

| Function | Description                                                  |
| -------- | ------------------------------------------------------------ |
| negList  | Negate all elements in a list using map                      |
| addLists | Add corresponding elements from two lists using zipWith      |
| sum      | Sum all elements in a list using reduce                      |
| prod     | Calculate the product of all elements in a list using reduce |
  
## Task 0.4: Modules  
This task is to implement the core structure of the minitorch.Module class. We ask you to implement a tree data structure that stores named minitorch.Parameter on each node. Such a data structure makes it easy for users to create trees that can be walked to find all of the parameters of interest.  
To experiment with the system use the Module Sandbox:  
```
streamlit run app.py -- 0

```
## Todo  
Complete the functions in minitorch/module.py and pass tests marked as task0_4.  
```
minitorch.Module.train() -> None

```
Set the mode of this module and all descendent modules to train.  
```
minitorch.Module.eval() -> None

```
Set the mode of this module and all descendent modules to eval.  
```
minitorch.Module.named_parameters() -> Sequence[Tuple[str, Parameter]]

```
Collect all the parameters of this module and its descendents.  
**Returns:** The name and Parameter of each ancestor parameter.  
**Returns:** The name and Parameter of each ancestor parameter.  
```
minitorch.Module.parameters() -> Sequence[Parameter]

```
Enumerate over all the parameters of this module and its descendents.  
  
## Task 0.5: Visualization  
For the first few assignments, we use a set of datasets implemented in minitorch/datasets.py, which are 2D point classification datasets. (See ++[TensorFlow Playground](https://playground.tensorflow.org/)++ for similar examples.) Each of these datasets can be added to the visualization.  
To experiment with the system use:  
```
streamlit run project/app.py -- 0

```
Read through the code in project/run_torch.py to get a sneak peek of an implementation of a model for these datasets using Torch.  
You can also provide a model that attempts to perform the classification by manipulating the parameters.  
## Todo  
* Add docstrings for all the different datasets required for this part.  
* Start a streamlit server and print an image of the dataset.  
* Hand-create classifiers that split the linear dataset into the correct colors.  
* Add the image in the README file in your repo along with the parameters that you used.  
