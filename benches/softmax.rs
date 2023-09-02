#[macro_use]
extern crate bencher;

use candle_core::{Device, Tensor};
use candle_nn;
use speedy_softmax::fused_softmax;

use bencher::Bencher;

const BATCH_SIZE: usize = 128;
const DIM: usize = 1024;

fn gen_batch(batch_size: usize, dim: usize) -> Tensor {
    Tensor::rand(-1.0f32, 1.0f32, &[batch_size, dim], &Device::Cpu).unwrap()
}

fn fused(bench: &mut Bencher) {
    let input = gen_batch(BATCH_SIZE, DIM);

    bench.iter(|| fused_softmax(&input, 1).unwrap());

    bench.bytes = (BATCH_SIZE * DIM * 4) as u64;
}

fn candle(bench: &mut Bencher) {
    let input = gen_batch(BATCH_SIZE, DIM);

    bench.iter(|| candle_nn::ops::softmax(&input, 1).unwrap());

    bench.bytes = (BATCH_SIZE * DIM * 4) as u64;
}

benchmark_group!(benches, candle, fused);
benchmark_main!(benches);
