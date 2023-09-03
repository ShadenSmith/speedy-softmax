use candle_core::{Device, Tensor};
use speedy_softmax::fused_softmax;

use criterion::*;

fn bench_exp_f32(c: &mut Criterion) {
    let batch_size = 128;
    let input = vec![1f32; batch_size];
    let mut output = vec![1f32; batch_size];
    let op_bytes = output.len() * 4 * 2;

    let mut group = c.benchmark_group("exponential");
    group.throughput(Throughput::Bytes(op_bytes as u64));
    group.bench_function("std::f32::exp", |b| {
        b.iter(|| {
            input
                .iter()
                .zip(output.iter_mut())
                .for_each(|(i, o)| *o = i.exp())
        });
    });
    group.finish();
}

fn bench_softmax_tensor(c: &mut Criterion) {
    let batch_size: usize = 1024;
    let hidden_dim: usize = 512;

    let batch = Tensor::rand(0f32, 1f32, &[batch_size, hidden_dim], &Device::Cpu).unwrap();

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

criterion_group!(benches, bench_exp_f32, bench_softmax_tensor);
criterion_main!(benches);
