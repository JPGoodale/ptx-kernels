# ptx-kernels
Some side by side comparisons of simple CUDA kernels written in pure inline PTX.


## Some notes on Naming Conventions:
In PTX assembly, virtual registers are usually denoted as `%r0` to %rN. However, when using inline PTX within CUDA code, directly using this notation can lead to name mangling issues because the inline assembly gets embedded into a larger .ptx file. To avoid conflicts with other parts of the code and ensure clear separation, the following naming conventions are adopted:

Temporary Registers: Instead of %r, we use %%t0 to %%tN, where t stands for a temporary register. The double % is necessary because the CUDA compiler automatically removes one % when generating the final PTX file. If only a single % were used, it would be removed, leading to syntax errors or undefined behaviors. In the generated PTX file, these registers will appear as %t0 to %tN.

Address Registers: Address registers, typically denoted as %rd in standard PTX, are replaced with %%ad. 

Predicate Registers: Predicate registers are usually represented as %p in PTX. In our inline assembly, they are denoted as %%pr.

## References 

https://docs.nvidia.com/cuda/parallel-thread-execution/index.html


https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
