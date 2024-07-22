#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    // Grid : (batch, head)

    // qkv offset is which_batch (batch_id * head * seq_len * head_dim) + which_head (head_id * seq_len * head_dim)
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    // lm_offset is batch_id * head * seq_len + head_id * seq_len
    // 当前批次在整个数据中的偏移量 + 前头在批次中的偏移量 = 序列在整个数据中的偏移位置。
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    // small size of the tile
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // flash attn 1, 先load kv、
        // tx：块内索引，确定块号
        // 找到 对应head offset之后，tile_size * j 确定是第几块。 tx 确定块内seq_len位置。x 确定head_dim位置
        // Load Kj, Vj to SRAM
        // TODO: 这里d如果写死应该可以循环展开，比如限定只能是64 128等
        // TODO: 考虑是否使用向量化访存
        for (int x = 0; x < d; x++) {
            // 每个线程在这里load进 2d的数据进share mem里
            // 同时有Bc个线程一起load，那么load进的数据就是 Bc * 2d。
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        // 同步一下，得到kv分块变量。
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // flash attn v1 后load Q，V2这里就改了
            // Load Qi to SRAM, l and m to registers
            // 和load KV一样，不赘述
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            // 每个线程维护一个自己的m和l
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // 一直load Q，不load KV，做矩阵乘法
            float row_m = -INFINITY;
            // 在for Bc 这个小row上每个thread统计一个局部的row_max
            // 每个thread 控制一行，for循环逐列做矩阵乘法。这里这么做是因为我们需要统计每一行的最大值。
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                // 在小块上做矩阵乘法
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            // 这里求Pij 和上述的也是一样
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            // 更新一组m 和 l
            // 找一个最大值
            float row_m_new = max(row_m_prev, row_m);
            // 用当前最大值和上一个最大值来更新row
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            // O写回 主存
            // l m 写回主存。因为要执行新的一段。要load 新的 l 和 m
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                // 中间结果 * V， 和上一个乘法的逻辑基本相同。
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        // 到这里，一个k块就执行完了，之后要做的就是继续遍历后续的kv块。完成整体的flash attn
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    // This should be setted to Bc = [M / 4d], Br = [M / 4d]
    const int Bc = 32; const int Br = 32;

    // B is batch, nh is num_head, N is seq_len, d is head_dim
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    // split KV seq_len to Tc and Q seq_len to Tr
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    // O is output
    // l and m is middle var
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    // each block deal with seq_len * head_dim range
    dim3 block_dim(Bc);  // Bc threads per block
    // each block deal with Bc * head_dim * 4 (Q and K an V and O, fill the share_mem)
    // each thread deal with head_dim * 4 elements

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
