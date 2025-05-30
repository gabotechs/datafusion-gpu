mod cubecl_sum_udaf;
#[cfg(feature = "cuda")]
mod cudarc_sum_udaf;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
use datafusion::arrow::array::{Float32Array, Int32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{CsvReadOptions, SessionConfig, SessionContext};
use rand::Rng;
use std::sync::Arc;

pub struct BuildCtxOpts {
    pub types_table_length: usize,
}

pub async fn build_ctx<R: Runtime>(opts: &BuildCtxOpts) -> anyhow::Result<SessionContext>
where
    <R as cubecl::Runtime>::Device: std::default::Default,
{
    let config = SessionConfig::new();
    let runtime = Arc::new(RuntimeEnv::default());
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .build();

    let compute_client = R::client(&Default::default());
    let ctx = SessionContext::new_with_state(state);

    let schema = Arc::new(Schema::new(vec![
        Field::new("string", DataType::Utf8, false),
        Field::new("float", DataType::Float32, false),
        Field::new("int", DataType::Int32, false),
    ]));

    let len = opts.types_table_length;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(generate_random_letters(len))),
            Arc::new(Float32Array::from(generate_random_numbers::<f32>(len))),
            Arc::new(Int32Array::from(generate_random_numbers::<i32>(len))),
        ],
    )?;

    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_udaf(cubecl_sum_udaf::udaf::<R>(Arc::new(compute_client)));
    #[cfg(feature = "cuda")]
    ctx.register_udaf(cudarc_sum_udaf::udaf(CudaDevice::new(0)?));
    ctx.register_table("types", Arc::new(table))?;
    ctx.register_csv("test", "datasets/test.csv", CsvReadOptions::default())
        .await?;
    Ok(ctx)
}

fn generate_random_letters(count: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| rng.gen_range('a'..='z').to_string())
        .collect()
}

fn generate_random_numbers<T>(count: usize) -> Vec<T>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen::<T>()).collect()
}
