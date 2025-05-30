use clap::Parser;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
#[cfg(not(feature = "cuda"))]
use cubecl::wgpu::WgpuRuntime;
use datafusion_gpu::{build_ctx, BuildCtxOpts};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokio::time::Instant;

#[derive(Parser)]
pub struct Args {
    #[arg(default_value = "")]
    pub sql: String,

    #[arg(short, long, default_value = "1024")]
    pub len: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let opts = BuildCtxOpts {
        types_table_length: args.len,
    };

    #[cfg(feature = "cuda")]
    let ctx = build_ctx::<CudaRuntime>(&opts).await?;
    #[cfg(not(feature = "cuda"))]
    let ctx = build_ctx::<WgpuRuntime>(&opts).await?;

    let mut rl = DefaultEditor::new()?;
    _ = rl.load_history(".history.txt");

    if !args.sql.is_empty() {
        let df = ctx.sql(&args.sql).await?;
        rl.add_history_entry(&args.sql)?;
        let start = Instant::now();
        df.show().await?;
        println!("Total execution time: {:?}", start.elapsed());
        return Ok(());
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
