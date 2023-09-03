use candle_core::{Device, Tensor};
use speedy_softmax::fused_softmax;

use criterion::*;

const BATCH_SIZE: usize = 1024;
const DIM: usize = 512;

fn bench_softmax(c: &mut Criterion) {
    let batch = Tensor::rand(0f32, 1f32, &[BATCH_SIZE, DIM], &Device::Cpu).unwrap();

    let op_bytes = batch.elem_count() * batch.dtype().size_in_bytes() * 2;

    let mut group = c.benchmark_group("softmax");
    group.throughput(Throughput::Bytes(op_bytes as u64));
    group.bench_function("candle", |b| {
        b.iter(|| candle_nn::ops::softmax(&batch, 1).unwrap())
    });
    group.bench_function("fused", |b| {
        b.iter(|| fused_softmax::softmax(&batch, 1).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_softmax);
criterion_main!(benches);
