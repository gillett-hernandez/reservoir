pub struct Reservoir<T, const N: usize> {
    pub data: [Option<T>; N],
    pub count: usize,
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
    pub fn gather_with_given_random_value(&mut self, sample: T, uniform01_sample: f32) {
        if self.count < N || uniform01_sample < (self.count as f32).recip() {
            self.data[self.count % N] = Some(sample);
        }
        self.count += 1;
    }
    #[cfg(feature = "rand")]
    pub fn gather(&mut self, sample: T) {
        self.gather_with_given_random_value(sample, rand::random::<f32>());
    }

    pub fn iter_values(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|e| e.as_ref())
    }
}
