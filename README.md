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
 
