pub mod candle;
pub mod fused_softmax;

#[cfg(test)]
mod tests {
    // Setup some testing utilities

    pub mod common {
        /// Round a float to 4 decimal places.
        pub fn round4(x: f32) -> f32 {
            (x * 10000.0).round() / 10000.0
        }
    }
}
