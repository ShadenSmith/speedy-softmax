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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::common::round4;

    #[test]
    fn test_softmax() {
        let xs: Vec<f32> = vec![
            -0.8414, 0.7612, -0.7832, 0.7465, -0.1200, -2.6573, 0.4870, -1.2291,
        ];

        let expected: Vec<f32> = vec![
            0.0538, 0.2671, 0.0570, 0.2632, 0.1107, 0.0088, 0.2030, 0.0365,
        ];

        let mut test = vec![0.; xs.len()];
        softmax_slice(&xs, &mut test);

        for (mine, gold) in test.into_iter().zip(expected) {
            assert_eq!(round4(mine), round4(gold));
        }
    }
}
