pub struct Reservoir<T, const N: usize> {
    data: [Option<T>; N],
    count: usize,
}
impl<T, const N: usize> Default for Reservoir<T, N> {
    fn default() -> Self {
        Self {
            data: [const { None }; N],
            count: 0,
        }
    }
}

impl<T, const N: usize> Reservoir<T, N> {
    pub fn record_with_random_value(&mut self, sample: T, uniform01_sample: f32) {
        if self.count < N || uniform01_sample < (self.count as f32).recip() {
            self.data[self.count % N] = Some(sample);
        }
        self.count += 1;
    }
    #[cfg(feature = "rand")]
    pub fn record(&mut self, sample: T) {
        self.record_with_random_value(sample, rand::random::<f32>());
    }

    pub fn iter_values(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|e| e.as_ref())
    }
}
