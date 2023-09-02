use candle_core::{cpu_backend::Map1, Device, Result, Tensor};
use clap::Parser;

use std::time::Instant;

use speedy_softmax::fused_softmax;

#[derive(Parser, Debug)]
struct Args {
    #[arg(default_value_t = 128)]
    /// Batch size of the input.
    batch_size: usize,

    #[arg(default_value_t = 1024)]
    /// Number of input features.
    hidden_dim: usize,

    #[arg(default_value_t = 1000)]
    /// Number of times to repeat the benchmark.
    reps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let gen_input = || {
        Tensor::rand(
            -1.0f32,
            1.0f32,
            &[args.batch_size, args.hidden_dim],
            &Device::Cpu,
        )
    };

    let input = gen_input()?;
    let start = Instant::now();
    for _ in 0..args.reps {
        let _ = candle_nn::ops::softmax(&input, 1)?;
    }
    let base_time = start.elapsed().as_secs_f64() / args.reps as f64;

    let input = gen_input()?;
    let start = Instant::now();
    for _ in 0..args.reps {
        let _ = fused_softmax(&input, 1)?;
    }
    let fused_time = start.elapsed().as_secs_f64() / args.reps as f64;

    println!(
        "Softmax: hidden: {}, batch: {}",
        args.hidden_dim, args.batch_size
    );
    println!("Base: {:.4} ms", base_time * 1000.0);
    println!("Fused: {:.4} ms", fused_time * 1000.0);
    println!("Speedup: {:.2}x", base_time / fused_time);

    Ok(())
}
