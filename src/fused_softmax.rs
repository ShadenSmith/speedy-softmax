use candle_core::CustomOp1;
use candle_core::{Result, Tensor};
use rayon::prelude::*;

#[cfg(not(feature = "fast-math"))]
fn exp_f32(x: f32) -> f32 {
    x.exp()
}

#[cfg(feature = "fast-math")]
fn exp_f32(x: f32) -> f32 {
    use fast_math::exp;
    exp(x)
}

pub fn softmax_slice(input: &[f32], output: &mut [f32]) {
    let sample_max = input.iter().copied().fold(f32::MIN, f32::max);

    let mut denominator = 0.0;
    input
        .iter()
        .zip(output.iter_mut())
        .for_each(|(in_val, out_val)| {
            *out_val = exp_f32(in_val - sample_max);
            denominator += *out_val;
        });

    denominator = 1.0 / denominator;

    output.iter_mut().for_each(|o| *o *= denominator);
}

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle_core::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = speedy_softmax::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    xs.apply_op1(FusedSoftmax { dim })
}

pub(crate) struct FusedSoftmax {
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
