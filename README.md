# machine-learning-papers

# MXNET Analysis

 # MXNET
 - Programming API
 - Gradient Calculation (Differentiation API)
 - Computational Graph Optimization and Execution
 - Runtime Paralle Scheduling 
 - GPU Kernels, Optimized Device Code
 - Accelerators and Hardware 

 ## Programming API
 Mxnet use Cpython to bind the C++ with python language 
 ## Gradient Calculation (Differentiation API) 
 [Computational Graph](http://colah.github.io/posts/2015-08-Backprop/) by Christopher Olah

 Key points:
 
 Forward-Mode differentiation: 
 
 Backward-Mode differentiation: We apply it to do the back propogation algorithm. Cuz we just care about taking derivative of loss function with respect to each node
 
 Details:
 http://dlsys.cs.washington.edu/pdf/lecture4.pdf
 
 Difference between computation graph and traditional back propogation algorithm:
 
 
 ## Computational Graph Optimization and Execution
 - Memory Planning 
 - PlaceDevice
 - InferShape
 http://dlsys.cs.washington.edu/pdf/lecture7.pdf
 ## Runtime Paralle Scheduling 
http://dlsys.cs.washington.edu/pdf/lecture9.pdf`
 ## GPU Kernels, Optimized Device Code
 Decouple the hardware related optimization from the computational graph
 http://dlsys.cs.washington.edu/pdf/lecture8.pdf
 ## Accelerators and Hardware 
 

# NNVM
http://dlsys.cs.washington.edu/pdf/lecture16.pdf

Training with Multiple GPUs Using Model Parallelism
http://mxnet.io/how_to/model_parallel_lstm.html


 
