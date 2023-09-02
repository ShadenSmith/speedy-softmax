pub mod fused_softmax;

use candle_core::{Result, Tensor};
pub fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    fused_softmax::fused_softmax(xs, dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::test_utils::to_vec2_round;
    use candle_core::Device;

    #[test]
    fn test_softmax() {
        let input = Tensor::rand(-1.0f32, 1.0f32, &[6, 34], &Device::Cpu).unwrap();

        let base_output = candle_nn::ops::softmax(&input, 1).unwrap();
        let test_output = softmax(&input, 1).unwrap();

        assert_eq!(
            to_vec2_round(&base_output, 4).unwrap(),
            to_vec2_round(&test_output, 4).unwrap(),
        );
    }
}
