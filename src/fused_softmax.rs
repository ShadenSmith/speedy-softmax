use candle_core::CustomOp1;
use candle_core::{Result, Tensor};

pub fn softmax_slice(xs: &[f32], out: &mut [f32]) {
    let sample_max = xs.iter().copied().fold(f32::MIN, f32::max);

    let mut denominator = 0f32;
    out.iter_mut().zip(xs.iter()).for_each(|(numerator, val)| {
        *numerator = (val - sample_max).exp();
        denominator += *numerator;
    });

    out.iter_mut().for_each(|x| *x /= denominator);
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

        let (dim1, dim2) = layout.shape().dims2()?;
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle_core::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };

        let mut dst: Vec<f32> = vec![0f32; dim1 * dim2];
        for idx1 in 0..dim1 {
            let sample = &src[idx1 * dim2..(idx1 + 1) * dim2];

            let mut out_row = &mut dst[idx1 * dim2..(idx1 + 1) * dim2];

            softmax_slice(sample, &mut out_row);
        }
        let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }
}
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
