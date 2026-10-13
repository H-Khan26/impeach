//! Synchronous retry with exponential backoff.
//!
//! Python origin: `utils/retry.py` (`RetryConfig` holds the settings, `RetryPolicy.run`
//! executes). Here [`RetryPolicy`] carries the settings and runs closures; sleeping and jitter
//! are injected through [`Sleeper`] and [`Jitter`] so tests never block.

use std::collections::hash_map::RandomState;
use std::fmt::Display;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Exponential backoff settings; field names and defaults match Python `RetryConfig`
/// (also the YAML keys under `rag.retry`). Durations are in seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetryPolicy {
    /// Retries happen only when enabled; otherwise the closure runs once. Default `false`.
    pub enabled: bool,
    /// Total attempts including the first; values below 1 mean 1. Default 3.
    pub max_attempts: u32,
    /// Delay before the first retry, seconds. Default 0.5.
    pub initial_backoff: f64,
    /// Upper bound on the delay, seconds; values <= 0 disable the bound. Default 4.0.
    pub max_backoff: f64,
    /// Factor applied to the delay after each retry; values <= 0 keep it constant. Default 2.0.
    pub backoff_multiplier: f64,
    /// Upper bound of the uniform random jitter added to each delay, seconds. Default 0.1.
    pub jitter: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_attempts: 3,
            initial_backoff: 0.5,
            max_backoff: 4.0,
            backoff_multiplier: 2.0,
            jitter: 0.1,
        }
    }
}

/// Blocks between attempts.
pub trait Sleeper {
    /// Waits for `duration`.
    fn sleep(&mut self, duration: Duration);
}

/// Sleeps the current thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadSleeper;

impl Sleeper for ThreadSleeper {
    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// Records requested sleeps without waiting; for tests.
#[derive(Debug, Clone, Default)]
pub struct RecordingSleeper {
    /// Every requested duration, in order.
    pub sleeps: Vec<Duration>,
}

impl Sleeper for RecordingSleeper {
    fn sleep(&mut self, duration: Duration) {
        self.sleeps.push(duration);
    }
}

/// Source of the random jitter added to each delay.
pub trait Jitter {
    /// Returns a value in `[0, max]` (Python `random.uniform(0, max)`).
    fn sample(&mut self, max: f64) -> f64;
}

/// Uniform pseudo-random jitter seeded from the process-random [`RandomState`].
#[derive(Debug, Clone, Default)]
pub struct RandomJitter {
    state: RandomState,
    counter: u64,
}

impl Jitter for RandomJitter {
    fn sample(&mut self, max: f64) -> f64 {
        let mut hasher = self.state.build_hasher();
        hasher.write_u64(self.counter);
        self.counter = self.counter.wrapping_add(1);
        let unit = (hasher.finish() >> 11) as f64 / (1_u64 << 53) as f64;
        unit * max
    }
}

/// Always returns the same fraction of `max`; for tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedJitter(pub f64);

impl Jitter for FixedJitter {
    fn sample(&mut self, max: f64) -> f64 {
        self.0 * max
    }
}

impl RetryPolicy {
    /// Number of attempts [`RetryPolicy::run_with`] makes at most: 1 when disabled, otherwise
    /// `max(1, max_attempts)`.
    #[must_use]
    pub fn effective_attempts(&self) -> u32 {
        if self.enabled {
            self.max_attempts.max(1)
        } else {
            1
        }
    }

    /// Runs `operation`, retrying every error with real sleeps and random jitter.
    ///
    /// # Errors
    /// Returns the last error once all attempts fail.
    pub fn run<T, E, F>(&self, description: &str, operation: F) -> Result<T, E>
    where
        E: Display,
        F: FnMut() -> Result<T, E>,
    {
        self.run_with(
            description,
            &mut ThreadSleeper,
            &mut RandomJitter::default(),
            |_: &E| true,
            operation,
        )
    }

    /// Runs `operation` until it succeeds, an error is not retryable, or attempts run out.
    ///
    /// With retries disabled or `max_attempts <= 1` the operation runs exactly once. Otherwise
    /// each retryable failure before the last attempt logs a warning and sleeps for
    /// `min(delay, max_backoff) + jitter` seconds (bound and jitter applied only when positive),
    /// then multiplies `delay` (starting at `max(0, initial_backoff)`) by `backoff_multiplier`
    /// and caps it at `max_backoff`. The final failure logs an error and is returned.
    /// Non-retryable errors return immediately without logging, like an exception outside
    /// Python's `retry_exceptions`.
    ///
    /// # Errors
    /// Returns the first non-retryable error or the last error once all attempts fail.
    pub fn run_with<T, E, F, R>(
        &self,
        description: &str,
        sleeper: &mut dyn Sleeper,
        jitter: &mut dyn Jitter,
        is_retryable: R,
        mut operation: F,
    ) -> Result<T, E>
    where
        E: Display,
        F: FnMut() -> Result<T, E>,
        R: Fn(&E) -> bool,
    {
        let attempts = self.effective_attempts();
        if attempts <= 1 {
            return operation();
        }
        let mut delay = self.initial_backoff.max(0.0);
        let mut attempt = 1;
        loop {
            let error = match operation() {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            if !is_retryable(&error) {
                return Err(error);
            }
            if attempt >= attempts {
                tracing::error!("{description} failed after {attempt} attempt(s): {error}");
                return Err(error);
            }
            let mut sleep_for = delay;
            if self.max_backoff > 0.0 {
                sleep_for = sleep_for.min(self.max_backoff);
            }
            if self.jitter > 0.0 {
                sleep_for += jitter.sample(self.jitter);
            }
            sleep_for = sleep_for.max(0.0);
            tracing::warn!(
                "{description} failed on attempt {attempt}/{attempts}: {error}; retrying in {sleep_for:.2}s"
            );
            sleeper.sleep(Duration::try_from_secs_f64(sleep_for).unwrap_or(Duration::MAX));
            if self.backoff_multiplier > 0.0 {
                delay *= self.backoff_multiplier;
            }
            if self.max_backoff > 0.0 {
                delay = delay.min(self.max_backoff);
            }
            attempt += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled(max_attempts: u32) -> RetryPolicy {
        RetryPolicy { enabled: true, max_attempts, ..RetryPolicy::default() }
    }

    /// Fails `failures` times, then succeeds with the number of calls.
    fn flaky(
        failures: u32,
    ) -> (impl FnMut() -> Result<u32, String>, std::rc::Rc<std::cell::Cell<u32>>) {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let counter = std::rc::Rc::clone(&calls);
        let op = move || {
            counter.set(counter.get() + 1);
            if counter.get() <= failures {
                Err(format!("failure {}", counter.get()))
            } else {
                Ok(counter.get())
            }
        };
        (op, calls)
    }

    fn assert_secs(actual: &[Duration], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a.as_secs_f64() - e).abs() < 1e-6, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn defaults_match_python_retry_config() {
        let policy = RetryPolicy::default();
        assert!(!policy.enabled);
        assert_eq!(policy.max_attempts, 3);
        assert!((policy.initial_backoff - 0.5).abs() < f64::EPSILON);
        assert!((policy.max_backoff - 4.0).abs() < f64::EPSILON);
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert!((policy.jitter - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn disabled_runs_once() {
        let (op, calls) = flaky(1);
        let mut sleeper = RecordingSleeper::default();
        let result = RetryPolicy::default().run_with(
            "op",
            &mut sleeper,
            &mut FixedJitter(0.0),
            |_| true,
            op,
        );
        assert_eq!(result, Err("failure 1".to_owned()));
        assert_eq!(calls.get(), 1);
        assert!(sleeper.sleeps.is_empty());
    }

    #[test]
    fn single_attempt_runs_once() {
        let (op, calls) = flaky(1);
        let mut sleeper = RecordingSleeper::default();
        let result = enabled(0).run_with("op", &mut sleeper, &mut FixedJitter(0.0), |_| true, op);
        assert!(result.is_err());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn retries_until_success_with_backoff() {
        let (op, calls) = flaky(2);
        let mut sleeper = RecordingSleeper::default();
        let result = enabled(3).run_with("op", &mut sleeper, &mut FixedJitter(0.0), |_| true, op);
        assert_eq!(result, Ok(3));
        assert_eq!(calls.get(), 3);
        assert_secs(&sleeper.sleeps, &[0.5, 1.0]);
    }

    #[test]
    fn gives_up_after_max_attempts() {
        let (op, calls) = flaky(10);
        let mut sleeper = RecordingSleeper::default();
        let result = enabled(3).run_with("op", &mut sleeper, &mut FixedJitter(0.0), |_| true, op);
        assert_eq!(result, Err("failure 3".to_owned()));
        assert_eq!(calls.get(), 3);
        assert_eq!(sleeper.sleeps.len(), 2);
    }

    #[test]
    fn delay_is_capped_and_jitter_added() {
        let (op, _) = flaky(10);
        let mut sleeper = RecordingSleeper::default();
        let _ = enabled(6).run_with("op", &mut sleeper, &mut FixedJitter(0.5), |_| true, op);
        // delay 0.5, 1, 2, 4, 4 (capped at max_backoff) plus half the 0.1 jitter.
        assert_secs(&sleeper.sleeps, &[0.55, 1.05, 2.05, 4.05, 4.05]);
    }

    #[test]
    fn non_positive_bounds_disable_cap_and_growth() {
        let policy = RetryPolicy {
            enabled: true,
            max_attempts: 4,
            initial_backoff: -1.0,
            max_backoff: 0.0,
            backoff_multiplier: 0.0,
            jitter: 0.0,
        };
        let (op, _) = flaky(10);
        let mut sleeper = RecordingSleeper::default();
        let _ = policy.run_with("op", &mut sleeper, &mut FixedJitter(1.0), |_| true, op);
        assert_eq!(sleeper.sleeps, vec![Duration::ZERO; 3]);

        let policy = RetryPolicy { initial_backoff: 3.0, backoff_multiplier: 2.0, ..policy };
        let (op, _) = flaky(10);
        let mut sleeper = RecordingSleeper::default();
        let _ = policy.run_with("op", &mut sleeper, &mut FixedJitter(1.0), |_| true, op);
        assert_secs(&sleeper.sleeps, &[3.0, 6.0, 12.0]);
    }

    #[test]
    fn non_retryable_error_returns_immediately() {
        let (op, calls) = flaky(10);
        let mut sleeper = RecordingSleeper::default();
        let result = enabled(3).run_with(
            "op",
            &mut sleeper,
            &mut FixedJitter(0.0),
            |e: &String| !e.ends_with('1'),
            op,
        );
        assert_eq!(result, Err("failure 1".to_owned()));
        assert_eq!(calls.get(), 1);
        assert!(sleeper.sleeps.is_empty());
    }

    #[test]
    fn run_succeeds_without_sleeping() {
        let (op, calls) = flaky(0);
        assert_eq!(enabled(3).run("op", op), Ok(1));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn random_jitter_stays_in_range() {
        let mut jitter = RandomJitter::default();
        for _ in 0..1000 {
            let value = jitter.sample(0.1);
            assert!((0.0..=0.1).contains(&value));
        }
    }

    #[test]
    fn deserializes_python_yaml_keys() {
        let policy: RetryPolicy =
            serde_json::from_str(r#"{"enabled": true, "max_attempts": 5, "jitter": 0.0}"#)
                .expect("valid retry json");
        assert_eq!(
            policy,
            RetryPolicy { enabled: true, max_attempts: 5, jitter: 0.0, ..RetryPolicy::default() }
        );
        assert!(serde_json::from_str::<RetryPolicy>(r#"{"retries": 2}"#).is_err());
    }
}
