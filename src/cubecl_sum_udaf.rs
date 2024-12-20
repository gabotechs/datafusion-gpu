use cubecl::prelude::*;
use cubecl_reduce::reduce_plane;
use cubecl_reduce::Sum as CubeSum;
use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::datatypes::{DataType, Field, Float32Type};
use datafusion::common::{exec_err, not_impl_err, Result, ScalarValue};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::utils::AggregateOrderSensitivity;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, ReversedUDAF, Signature,
};
use delegate::delegate;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

pub fn udaf<R: Runtime>(compute_client: Arc<ComputeClient<R::Server, R::Channel>>) -> AggregateUDF {
    AggregateUDF::from(GpuSum::<R> {
        sum: Sum::default(),
        compute_client,
    })
}

#[derive(Debug)]
struct GpuSum<R: Runtime> {
    sum: Sum,
    compute_client: Arc<ComputeClient<R::Server, R::Channel>>,
}

impl<R: Runtime> AggregateUDFImpl for GpuSum<R> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "sum_cubecl"
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let c = self.compute_client.clone();
        let t = args.exprs[0].data_type(args.schema)?;
        match t {
            DataType::Int32 => Ok(Box::new(GpuSumAccumulator::<R>::new(c))),
            DataType::UInt32 => Ok(Box::new(GpuSumAccumulator::<R>::new(c))),
            DataType::Float32 => Ok(Box::new(GpuSumAccumulator::<R>::new(c))),
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

        Ok(vec![match &arg_types[0] {
            dt if dt.is_signed_integer() => DataType::Int32,
            dt if dt.is_unsigned_integer() => DataType::UInt32,
            dt if dt.is_floating() => DataType::Float32,
            _ => return exec_err!("Sum not supported for {}", arg_types[0]),
        }])
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

struct GpuSumAccumulator<R: Runtime> {
    compute_client: Arc<ComputeClient<R::Server, R::Channel>>,
    // TODO: Having a generic number here is very difficult, I still don't know how to do it, so I just use f32.
    result: f32,
}

unsafe impl<R: Runtime> Send for GpuSumAccumulator<R> {}
unsafe impl<R: Runtime> Sync for GpuSumAccumulator<R> {}

impl<R: Runtime> GpuSumAccumulator<R> {
    fn new(compute_client: Arc<ComputeClient<R::Server, R::Channel>>) -> Self {
        Self {
            compute_client,
            result: 0.0,
        }
    }
}

impl<R: Runtime> Debug for GpuSumAccumulator<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuSumAccumulator(?)",)
    }
}

#[cube(launch_unchecked)]
fn sum<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    reduce_dim: u32,
    #[comptime] cube_dim: u32,
    #[comptime] exact_shape: bool,
) {
    reduce_plane::<CubeSum, N, N>(input, output, reduce_dim, cube_dim, exact_shape)
}

impl<R: Runtime> Accumulator for GpuSumAccumulator<R> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let data_type = values[0].data_type();
        let data = values[0].to_data();
        let len = values[0].len();
        let data: &[u8] = data.buffers().first().map(|b| b.as_slice()).unwrap();

        let block_size = 1024;
        let num_blocks = len.div_ceil(block_size);

        let output_handle = match data_type {
            DataType::Int32 => self.compute_client.empty(size_of::<i32>()),
            DataType::UInt32 => self.compute_client.empty(size_of::<u32>()),
            DataType::Float32 => self.compute_client.empty(size_of::<f32>()),
            v => return not_impl_err!("SumGpu not supported for {}", v),
        };
        let input_handle = self.compute_client.create(data);

        // TODO: Why this can only be 1?
        const LINE_SIZE: u8 = 1;
        unsafe {
            macro_rules! run {
                ($ty: ty) => {
                    sum::launch_unchecked::<$ty, R>(
                        self.compute_client.as_ref(),
                        CubeCount::new_1d(num_blocks as u32),
                        CubeDim::new_1d(block_size as u32),
                        TensorArg::from_raw_parts::<$ty>(&input_handle, &[1], &[len], LINE_SIZE),
                        TensorArg::from_raw_parts::<$ty>(&output_handle, &[1], &[1], LINE_SIZE),
                        ScalarArg::new(0),
                        block_size as u32,
                        false, // TODO: calc if the plane dim has the exact size of the data.
                    )
                };
            }

            match data_type {
                DataType::Int32 => run!(i32),
                DataType::UInt32 => run!(u32),
                DataType::Float32 => run!(f32),
                v => return not_impl_err!("SumGpu not supported for {}", v),
            }
        }

        let mut bytes = self
            .compute_client
            .read(vec![output_handle.clone().binding()]);
        let bytes = bytes.remove(0);

        macro_rules! dump {
            ($ty: ty) => {
                for v in <$ty>::from_bytes(&bytes) {
                    self.result += *v as f32;
                }
            };
        }

        match data_type {
            DataType::Int32 => dump!(i32),
            DataType::UInt32 => dump!(u32),
            DataType::Float32 => dump!(f32),
            v => return not_impl_err!("SumGpu not supported for {}", v),
        };
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
