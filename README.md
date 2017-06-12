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
 
 Execute the computational graph:
 https://github.com/dmlc/mxnet/blob/986b736b816018b96e9d1e2c358bb7665b80944d/src/executor/graph_executor.cc#L51
 
 http://dlsys.cs.washington.edu/pdf/lecture7.pdf
 ## Runtime Paralle Scheduling 
http://dlsys.cs.washington.edu/pdf/lecture9.pdf`
 ## GPU Kernels, Optimized Device Code
 Decouple the hardware related optimization from the computational graph
 http://dlsys.cs.washington.edu/pdf/lecture8.pdf
 ## Accelerators and Hardware 
 
# Mshadow
General operation:
https://github.com/dmlc/mshadow/tree/master/guide
Cutting-edge techniques:
https://github.com/dmlc/mshadow/tree/master/guide/exp-template
https://en.wikipedia.org/wiki/Expression_templates

# NNVM
http://dlsys.cs.washington.edu/pdf/lecture16.pdf

### NNVM operator
https://github.com/dmlc/nnvm/blob/master/include/nnvm/op.h


### Connect the front-end to the back-end
https://github.com/dmlc/mxnet/tree/master/src/c_api
https://github.com/dmlc/mxnet/blob/master/src/c_api/c_api_symbolic.cc#L93
https://github.com/dmlc/nnvm/blob/master/src/core/symbolic.cc#L509

Training with Multiple GPUs Using Model Parallelism
http://mxnet.io/how_to/model_parallel_lstm.html

# Examples
https://github.com/yuruofeifei/assignment1/blob/master/autodiff_test.py
https://github.com/yuruofeifei/assignment1/blob/master/autodiff.py

# MShadow
# CUDA
http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

Stream: A sequence of operations that execute in issue-order on the GPU

Stride: this is used to deal with pitch allocation in GPU or SSE (align x dimension to 64bit) for efficiency

Pitch memory: aligned location for GPU

Requirements: Data used by concurrent operations should be independent

#program unroll

https://stackoverflow.com/questions/22278631/what-does-pragma-unroll-do-exactly-does-it-affect-the-number-of-threads

cudaGetDeviceCount(int* count);
return the number of devices with compute capabilities greater or equal to 1.0

__host__ cudaError_t cudaSetDevice ( int  device )
Set device to be used for GPU executions.

_host__ cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )
Returns information about the compute-device.



# Reference
### MXNet System Architecture
http://mxnet.io/architecture/overview.html

### Deep Learning Programming Style
http://mxnet.io/architecture/program_model.html

### Dependency Engine for Deep Learning
http://mxnet.io/architecture/note_engine.html

### Optimizing Memory Consumption in Deep Learning
http://mxnet.io/architecture/note_memory.html


Tinyflow .. To be continued
 
