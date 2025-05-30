#![cfg(feature = "cuda")]

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::cuda::CudaRuntime;
use datafusion_gpu::{build_ctx, BuildCtxOpts};
use std::sync::Arc;

pub fn cuda_sum(c: &mut Criterion) {
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap(),
    );

    let ctx = tokio::sync::OnceCell::new();

    c.bench_function("cuda_sum", move |b| {
        let rt = rt.clone();
        b.to_async(rt.as_ref()).iter(|| async {
            let ctx = ctx
                .get_or_init(|| async {
                    build_ctx::<CudaRuntime>(&BuildCtxOpts {
                        types_table_length: 100000,
                    })
                    .await
                    .unwrap()
                })
                .await;

            ctx.sql("SELECT sum_cudarc(float) FROM types")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().configure_from_args().sample_size(50);
    targets = cuda_sum,
}
criterion_main!(benches);
