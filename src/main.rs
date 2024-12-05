mod cubecl_sum_udaf;
#[cfg(feature = "cuda")]
mod cudarc_sum_udaf;

use clap::Parser;
use cubecl::Runtime;
use datafusion::arrow::array::{Float32Array, Int32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{CsvReadOptions, SessionConfig, SessionContext};
use rand::Rng;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(not(feature = "cuda"))]
use cubecl::wgpu::WgpuRuntime;
use tokio::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(default_value = "")]
    sql: String,

    #[arg(short, long, default_value = "1024")]
    len: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    #[cfg(feature = "cuda")]
    let ctx = build_ctx::<CudaRuntime>(&args).await?;
    #[cfg(not(feature = "cuda"))]
    let ctx = build_ctx::<WgpuRuntime>(&args).await?;

    let mut rl = DefaultEditor::new()?;
    _ = rl.load_history(".history.txt");

    if !args.sql.is_empty() {
        let df = ctx.sql(&args.sql).await?;
        rl.add_history_entry(&args.sql)?;
        let start = Instant::now();
        df.show().await?;
        println!("Total execution time: {:?}", start.elapsed());
    }

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                rl.save_history(".history.txt")?;
                if line == "q" || line == "exit" || line == "quit" {
                    return Ok(());
                }
                let df = match ctx.sql(&line).await {
                    Ok(v) => v,
                    Err(err) => {
                        println!("{}", err);
                        continue;
                    }
                };

                let start = Instant::now();
                df.show().await?;
                println!("Total execution time: {:?}", start.elapsed());
                println!()
            }

            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                return Ok(());
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                return Ok(());
            }
            Err(err) => return Err(err.into()),
        };
    }
}

async fn build_ctx<R: Runtime>(args: &Args) -> anyhow::Result<SessionContext>
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

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(generate_random_letters(args.len))),
            Arc::new(Float32Array::from(generate_random_numbers::<f32>(args.len))),
            Arc::new(Int32Array::from(generate_random_numbers::<i32>(args.len))),
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
    (0..count).map(|_| rng.gen_range('a'..='z').to_string()).collect()
}

fn generate_random_numbers<T>(count: usize) -> Vec<T>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen::<T>()).collect()
}

