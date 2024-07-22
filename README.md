# flash-attention-minimal中文增强版

## 现在有什么问题

没有反向传播！老实说，我发现反向传播比前向传播复杂得多，而前向传播已经足够展示如何使用共享内存来避免大量的 N^2 读/写操作。

在内循环中，我将每个线程分配给输出矩阵的一行。这与原始实现有所不同。

这种每线程一行的简化使得矩阵乘法非常慢。这可能是为什么对于较长的序列和较大的块大小，这比手动实现要慢。

Q、K、V 是 float32，不像原始实现中使用的 float16。

块大小在编译时固定为 32。


## 在之前的基础上我们要做什么？
- [x] 添加代码阅读注释
- [ ] 尝试实现一个fp16的版本
- [ ] 尝试修改某些限制，比如很慢的矩阵乘法。
- [ ] 实现一个可变的QKV块
- [ ] 添加反向传播
 
————————————————————origin————————————————————————————

A minimal re-implementation of Flash Attention with CUDA and PyTorch. 
The official [implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner
(like myself), so this repo tries to be small and educational.

* The entire forward pass is written in ~100 lines in `flash.cu`.
* The variable names follow the notations from the original [paper](https://arxiv.org/abs/2205.14135).

## Usage
### Prerequisite
* PyTorch (with CUDA)
* `Ninja` for loading in C++

### Benchmark
Compare the wall-clock time between manual attention and minimal flash attention:
```
python bench.py
```

Sample output on a [T4](https://aws.amazon.com/ec2/instance-types/g4/):
```
=== profiling manual attention ===
...
Self CPU time total: 52.389ms
Self CUDA time total: 52.545ms

=== profiling minimal flash attention === 
...  
Self CPU time total: 11.452ms
Self CUDA time total: 3.908ms
```
Speed-up achieved! 


## Caveats
* No backward pass! To be honest, I found it a lot more complex than the forward pass, which was enough to show the
use of shared memory to avoid large N^2 read/writes.
* In the inner loop, I assign each thread to a row of the output matrix. This differs from the original implementation.
* This thread-per-row simplification makes the matrix multiplications very slow. This is probably why for longer 
sequences and larger block sizes, this gets slower than the manual implementation.
* Q,K,Vs are in float32, unlike the original implementation which uses float16.
* The block size is [fixed](https://github.com/tspeterkim/flash-attention-minimal/blob/9b7ca8ef4e6afdbfeb149a9cd488c8dea9af9ad6/flash.cu#L85) at compile time to 32.
