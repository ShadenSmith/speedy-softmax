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
