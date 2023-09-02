use candle_core::{Device, Result, Tensor};
use clap::Parser;

use speedy_softmax::fused_softmax;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;

    let baseline = candle_nn::ops::softmax(&input, 1)?;

    println!("{input}");
    let output = fused_softmax(&input, 1)?;
    println!("{output}");
    Ok(())
}
