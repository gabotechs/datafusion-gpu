# Datafusion GPU

This repo intends to showcase the capabilities of wiring up [Apache Datafusion](https://github.com/apache/datafusion)
with GPU execution runtimes in order to speed up heavy computations.

## Objective

The main objective is not to provide a wide set of execution nodes or math functions that can
run on GPU, but trying out different technologies to run a single aggregation function, and see
what are the benefits and drawbacks of each approach.

For that, two approaches where followed:

## Compiling compute kernels at runtime with [CubeCL](https://github.com/tracel-ai/cubecl)

This approach uses https://github.com/tracel-ai/cubecl for writing kernels directly in Rust
code, which get compiled down to different backends, like CUDA or WGPU.

[Example here](./src/cubecl_sum_udaf.rs)

### Advantages

- Write the kernel once, and use it for any datatype and several GPU technologies
- Use Rust for writing the kernel, no need to learn hardware-specific languages

### Disadvantages

- Small ecosystem, lack of documentation, lack of examples, immature technology
- Bad performance (this could be on me)
- Bugs? (got my laptop bricked several times trying to run some kernels)
- Certain abstractions very tailored to working with Tensors rather than 1d arrays

## Writing CUDA kernels by hand and feeding them data with [cudarc](https://github.com/coreylowman/cudarc)

This approach uses https://github.com/coreylowman/cudarc with some handwritten CUDA kernels. The
library allows feeding buffers to the GPU and scheduling them to be executed using the compiled
handwritten kernel.

[Example here](./src/cudarc_sum_udaf.rs)

### Advantages

- More control over the kernel code that gets executed on the GPU
- Good performance
- Wide CUDA ecosystem

### Disadvantages

- Works only on CUDA devices
- Making a kernel work for different datatypes is not supported out of the box
- Needs knowledge about writing CUDA code

## Results

Given the following conditions:

- Measured on a g4dn.xlarge AWS instance with 4vCPU and a T4 GPU
- In-memory table called `types` with 1000000 entries with the following schema:

```sql
+--------+-------+-----+
| string | float | int |
+--------+-------+-----+
```

- Datafusion runtime with two `sum` aggregation function variants:
    - `sum_cubecl`: using plane reduction kernel from [cudarc](https://github.com/coreylowman/cudarc)
    - `sum_cudarc`: using a handwritten CUDA kernel based on a shared memory algorithm
- Code run with the following command: `cargo run --release --features cuda -- -l 1000000`

| Query                               | Execution time |
|-------------------------------------|----------------|
| SELECT sum(float) FROM types        | ~7.5ms         |
| SELECT sum_cudarc(float) FROM types | ~2ms           |
| SELECT sum_cubecl(float) FROM types | ~440ms         |
