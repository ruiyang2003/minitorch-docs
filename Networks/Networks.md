# Networks  
We now have a fully working deep learning library with most of the features of a real industrial system like Torch. To take advantage of this hard work, this module is entirely based on using the software framework. In particular, we are going to build an image recognition system. We will do this by building the infrastructure for a version of LeNet on MNIST: a classic convolutional neural network (CNN) for digit recognition, and for a 1D conv for NLP sentiment classification.  
You need the files from previous assignments, so make sure to pull them over to your new repo. We recommend you get familiar with tensor.py, since you might find some of those functions useful for implementing this Module.  
  
## Guides  
* Convolution  
* Pooling  
* Softmax  
  
## Task 4.1: 1D Convolution  
You will implement the 1D convolution in Numba. This function gets used by the forward and backward pass of conv1d.  
## Todo  
Complete the following function in minitorch/fast_conv.py, and pass tests marked as task4_1.  
* _tensor_conv1d  
  
## Task 4.2: 2D Convolution  
You will implement the 2D convolution in Numba. This function gets used by the forward and backward pass of conv2d.  
## Todo  
Complete the following function in minitorch/fast_conv.py, and pass tests marked as task4_2.  
* _tensor_conv2d  
  
## Task 4.3: Pooling  
You will implement 2D pooling on tensors with an average operation.  
## Todo  
Complete the following function in minitorch/nn.py, and pass tests marked as task4_3. Use it to implement avgpool2d.  
```
minitorch.tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]

```
Reshape an image tensor for 2D pooling.  
**Input:**  
```
input: batch x channel x height x width
kernel: height x width of pooling

```
**Returns:**  
```
Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width)
as well as the new_height and new_width value.

```
  
## Task 4.4: Softmax and Dropout  
You will implement max, softmax, and log softmax on tensors as well as the dropout and max-pooling operations.  
## Todo  
* Complete the following functions in minitorch/nn.py, and pass tests marked as task4_4.  
* Add property tests for the functions in test/test_nn.py and ensure that you understand their gradient computation.  

| Function             | Description                 |
| -------------------- | --------------------------- |
| minitorch.max        | Max operation over a tensor |
| minitorch.softmax    | Softmax over a tensor       |
| minitorch.logsoftmax | Log softmax over a tensor   |
| minitorch.maxpool2d  | 2D max pooling              |
| minitorch.dropout    | Dropout operation           |
  
## Task 4.4b: Extra Credit  
Implementing convolution and pooling efficiently is critical for large-scale image recognition. However, both are a bit harder than some of the basic CUDA functions we have written so far. For this task, add an extra file cuda_conv.py that implements conv1d and conv2d on CUDA. Show the output on Colab.  
  
## Task 4.5: Training an Image Classifier  
If your code works, you should now be able to move on to the NLP and CV training scripts in project/run_sentiment.py and project/run_mnist_multiclass.py. This script has the same basic training setup as Module 3, but now adapted to sentiment and image classification. You need to implement Conv1D, Conv2D, and Network for both files.  
We recommend running on the command line when testing. But you can also use the Streamlit visualization to view hidden states of your model.  
## Todo  
* Train a model on **Sentiment (SST2)**, and add your training printout logs as a text file sentiment.txt to the repo. It should show train loss, train accuracy, and validation accuracy. *(The model should reach >70% best validation accuracy.)*  
* Train a model on **Digit Classification (MNIST)**, and add your training printout logs as a text file mnist.txt to the repo. It should show train loss and validation accuracy out of 16.  
