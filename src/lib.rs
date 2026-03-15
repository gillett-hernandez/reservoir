//! # Reservoir
//!
//! `reservoir` is a simple reservoir sampling crate. use the `rand` feature to enable the simple `gather` method, or use your own rng

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

impl<T> Reservoir<T, 1> {
    /// panics if the reservoir does not contain anything
    pub fn unwrap(&self) -> &T {
        self.data[0].as_ref().unwrap()
    }
}

impl<T, const N: usize> Reservoir<T, N> {
    /// precondition: rng must be a closure that produces a random number between 0 and the passed in argument
    /// something that causes tests to pass (and is implemented if the `rand` feature is enabled) is rand::random_in_range
    pub fn gather_precise<F: FnMut(usize) -> usize>(&mut self, sample: T, mut rng: F) {
        if self.count < N {
            self.data[self.count] = Some(sample);
            self.count += 1;
        } else {
            let index = rng(self.count);
            if index < N {
                self.data[index] = Some(sample);
            }
            self.count += 1;
        }
    }
    /// Requires the `rand` feature to be enabled
    #[cfg(feature = "rand")]
    pub fn gather(&mut self, sample: T) {
        self.gather_precise(sample, |c| rand::random_range(0..c + 1));
    }

    pub fn iter_values(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|e| e.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    /// Minimal xorshift64 PRNG — produces uniform floats in [0, 1) with no extra dependencies.
    struct Xorshift(u64);
    impl Xorshift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // top 24 bits → [0, 1)
            (self.0 >> 40) as f32 / 16_777_216.0
        }
        fn rand_in_range(&mut self, range: Range<usize>) -> usize {
            let delta = range.end - range.start;
            let chosen = (delta as f32 * self.next_f32()) as usize;
            chosen + range.start
        }
    }

    /// Statistical test: run many independent reservoir trials over a known population and
    /// verify that each item is selected with the expected uniform probability N/POPULATION.
    ///
    /// This is equivalent to asking: "does streaming reservoir sampling produce the same
    /// distribution as drawing N items uniformly at random without replacement at the end?"
    /// The chi-squared goodness-of-fit test answers this at a significance level of ~0.001.
    #[test]
    fn reservoir_matches_uniform_without_replacement() {
        const N: usize = 10;
        const POPULATION: usize = 100;
        const TRIALS: usize = 20_000;

        let mut counts = [0u32; POPULATION];

        for _ in 0..TRIALS {
            let mut r = Reservoir::<usize, N>::default();
            for item in 0..POPULATION {
                r.gather_precise(item, |c| {
                    if c == 0 {
                        0
                    } else {
                        rand::random_range(0..c + 1)
                    }
                });
            }
            for val in r.iter_values() {
                counts[*val] += 1;
            }
        }

        // Under uniform sampling without replacement each item is chosen with probability
        // N/POPULATION, so expected count per item = TRIALS * N / POPULATION = 2000.
        let expected = TRIALS as f64 * N as f64 / POPULATION as f64;

        // Chi-squared goodness-of-fit statistic with df = POPULATION - 1 = 99.
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();

        // Critical value for χ²(df=99) at significance 0.001 ≈ 148.2.
        // A threshold of 160 gives robustness against rare false failures while still
        // detecting any meaningful departure from uniformity.
        assert!(
            chi_sq < 160.0,
            "chi-squared = {chi_sq:.2} (df=99, critical ≈ 148.2); \
             reservoir distribution appears non-uniform"
        );
    }

    #[test]
    fn reservoir_matches_uniform_without_replacement2() {
        const N: usize = 1;
        const POPULATION: usize = 100;
        const TRIALS: usize = 20_000;

        let mut counts = [0u32; POPULATION];

        for _ in 0..TRIALS {
            let mut r = Reservoir::<usize, N>::default();
            for item in 0..POPULATION {
                r.gather_precise(item, |c| {
                    if c == 0 {
                        0
                    } else {
                        rand::random_range(0..c + 1)
                    }
                });
            }
            for val in r.iter_values() {
                counts[*val] += 1;
            }
        }

        // Under uniform sampling without replacement each item is chosen with probability
        // N/POPULATION, so expected count per item = TRIALS * N / POPULATION = 2000.
        let expected = TRIALS as f64 * N as f64 / POPULATION as f64;

        // Chi-squared goodness-of-fit statistic with df = POPULATION - 1 = 99.
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();

        // Critical value for χ²(df=99) at significance 0.001 ≈ 148.2.
        // A threshold of 160 gives robustness against rare false failures while still
        // detecting any meaningful departure from uniformity.
        assert!(
            chi_sq < 160.0,
            "chi-squared = {chi_sq:.2} (df=99, critical ≈ 148.2); \
             reservoir distribution appears non-uniform"
        );
    }

    #[test]
    fn reservoir_matches_uniform_without_replacement3() {
        const N: usize = 10;
        const POPULATION: usize = 100;
        const TRIALS: usize = 20_000;

        let mut counts = [0u32; POPULATION];
        let mut rng = Xorshift(0xdeadbeef_cafebabe);

        for _ in 0..TRIALS {
            let mut r = Reservoir::<usize, N>::default();
            for item in 0..POPULATION {
                r.gather_precise(item, |c| rng.rand_in_range(0..c + 1));
            }
            for val in r.iter_values() {
                counts[*val] += 1;
            }
        }

        // Under uniform sampling without replacement each item is chosen with probability
        // N/POPULATION, so expected count per item = TRIALS * N / POPULATION = 2000.
        let expected = TRIALS as f64 * N as f64 / POPULATION as f64;

        // Chi-squared goodness-of-fit statistic with df = POPULATION - 1 = 99.
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();

        // Critical value for χ²(df=99) at significance 0.001 ≈ 148.2.
        // A threshold of 160 gives robustness against rare false failures while still
        // detecting any meaningful departure from uniformity.
        assert!(
            chi_sq < 160.0,
            "chi-squared = {chi_sq:.2} (df=99, critical ≈ 148.2); \
             reservoir distribution appears non-uniform"
        );
    }

    /// Sanity check: the reservoir must hold exactly N items after processing more than N items.
    #[test]
    fn reservoir_fills_to_capacity() {
        const N: usize = 5;
        let mut r = Reservoir::<i32, N>::default();
        let mut rng = Xorshift(0x1234567890abcdef);
        for i in 0..20i32 {
            r.gather_precise(i, |c| rng.rand_in_range(0..c + 1));
        }
        assert_eq!(r.iter_values().count(), N);
    }
}
