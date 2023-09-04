use candle_core::CustomOp1;
use candle_core::{Result, Tensor};

use crate::fused_softmax::softmax_slice;

use rayon::prelude::*;

pub fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    xs.apply_op1(FusedSoftmax { dim })
}

pub struct FusedSoftmax {
    /// The dimension along which to compute the softmax.
    pub dim: usize,
}

impl CustomOp1 for FusedSoftmax {
    fn name(&self) -> &'static str {
        "fused-softmax"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        if self.dim != 1 {
            candle_core::bail!("only dim=1 is supported");
        }

        // Lower the input tensor to an f32 slice
        let (batch_size, dim) = layout.shape().dims2()?;
        let batch = match layout.contiguous_offsets() {
            None => candle_core::bail!("speedy-softmax input must be contiguous"),
            Some((start, end)) => {
                let slice = storage.as_slice::<f32>()?;
                &slice[start..end]
            }
        };

        let mut output: Vec<f32> = vec![0f32; batch_size * dim];

        // Compute the softmax, threading over the batch dimension
        output
            .par_chunks_exact_mut(dim)
            .zip(batch.par_chunks_exact(dim))
            .for_each(|(out, inp)| softmax_slice(inp, out));

        let storage = candle_core::WithDType::to_cpu_storage_owned(output);
        Ok((storage, layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::test_utils::to_vec2_round;
    use candle_core::Device;

    #[test]
    fn test_fused_softmax() {
        let input = Tensor::rand(-1.0f32, 1.0f32, &[6, 10], &Device::Cpu).unwrap();

        let base_output = candle_nn::ops::softmax(&input, 1).unwrap();
        let test_output = softmax(&input, 1).unwrap();

        assert_eq!(
            to_vec2_round(&base_output, 5).unwrap(),
            to_vec2_round(&test_output, 5).unwrap(),
        );
    }
}
