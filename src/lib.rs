use candle_core::{Device, Result, Tensor};

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle_core::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = speedy_softmax::fused_softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn fused_softmax<D: candle_core::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "fused-softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::test_utils::to_vec2_round;

    #[test]
    fn test_fused_softmax() {
        let input = Tensor::rand(-1.0f32, 1.0f32, &[6, 34], &Device::Cpu).unwrap();

        let base_output = candle_nn::ops::softmax(&input, 1).unwrap();
        let test_output = fused_softmax(&input, 1).unwrap();

        assert_eq!(
            to_vec2_round(&base_output, 4).unwrap(),
            to_vec2_round(&test_output, 4).unwrap(),
        );
    }
}
