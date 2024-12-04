use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx};
use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::datatypes::{DataType, Field, Float32Type};
use datafusion::common::{exec_err, not_impl_err, Result, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::utils::AggregateOrderSensitivity;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, ReversedUDAF, Signature,
};
use delegate::delegate;
use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

// language=cu
const PTX_SRC: &str = "
extern \"C\" __global__ void sum(float* input, float* result, int size) {
    // Shared memory for storing partial sums
    extern __shared__ float sharedData[];

    // Thread index within the block
    int threadId = threadIdx.x;
    // Global index in the array
    int globalIndex = blockIdx.x * blockDim.x + threadId;

    // Load elements into shared memory
    if (globalIndex < size) {
        sharedData[threadId] = input[globalIndex];
    } else {
        sharedData[threadId] = 0.0f;  // Handle out-of-bounds threads
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (threadId == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}
";

pub fn udaf(dev: Arc<CudaDevice>) -> AggregateUDF {
    let ptx = compile_ptx(PTX_SRC).unwrap();
    dev.load_ptx(ptx, "sum", &["sum"]).unwrap();
    AggregateUDF::from(GpuSum {
        sum: Sum::default(),
        dev,
    })
}

#[derive(Debug)]
struct GpuSum {
    sum: Sum,
    dev: Arc<CudaDevice>,
}

impl AggregateUDFImpl for GpuSum {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "sum_cudarc"
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let t = args.exprs[0].data_type(args.schema)?;
        match t {
            DataType::Float32 => Ok(Box::new(GpuSumAccumulator::new(self.dev.clone()))),
            v => {
                not_impl_err!("SumGpu not supported for {}: {}", args.name, v)
            }
        }
    }

    fn create_sliding_accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        // TODO: does the same accumulator work?
        self.accumulator(args)
    }

    // This function cannot be delegated to self.sum, because it will mess
    // with the input types.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 1 {
            return exec_err!("SUM expects exactly one argument");
        }

        Ok(vec![DataType::Float32])
    }

    delegate! {
        to self.sum {
            fn signature(&self) -> &Signature;
            fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Field>>;
            // TODO: groups accumulators are not supported.
            // fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool;
            // fn create_groups_accumulator( &self, args: AccumulatorArgs) -> Result<Box<dyn GroupsAccumulator>>;
            fn aliases(&self) -> &[String];
            fn order_sensitivity(&self) -> AggregateOrderSensitivity;
            fn reverse_expr(&self) -> ReversedUDAF;
        }
    }
}

struct GpuSumAccumulator {
    result: f32,
    dev: Arc<CudaDevice>,
    f: CudaFunction,
}

impl GpuSumAccumulator {
    fn new(dev: Arc<CudaDevice>) -> Self {
        let f = dev.get_func("sum", "sum").unwrap();
        Self {
            result: 0.0,
            dev,
            f,
        }
    }
}

impl Debug for GpuSumAccumulator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuSumAccumulator(?)",)
    }
}

fn df_err<T: Display>(err: T) -> DataFusionError {
    DataFusionError::Execution(err.to_string())
}

impl Accumulator for GpuSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let size = values[0].len();
        let block_size = 256;
        let num_blocks = (size + block_size - 1) / block_size;
        let shared_mem = block_size * size_of::<f32>();

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32
        };

        let data = values[0].to_data();
        let in_host = data.buffer::<f32>(0);
        let in_dev = self.dev.htod_sync_copy(&in_host).map_err(df_err)?;
        let mut out_host = vec![0.0f32; num_blocks];
        let mut out_dev = self.dev.htod_sync_copy(&out_host).map_err(df_err)?;

        unsafe { self.f.clone().launch(cfg, (&in_dev, &mut out_dev, size)) }.map_err(df_err)?;

        self.dev
            .dtoh_sync_copy_into(&out_dev, &mut out_host)
            .map_err(df_err)?;
        for v in out_host {
            self.result += v;
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        ScalarValue::new_primitive::<Float32Type>(Some(self.result), &DataType::Float32)
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}
