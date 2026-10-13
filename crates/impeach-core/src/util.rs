//! Small helpers shared by every crate: poison-tolerant locking, saturating integer
//! conversion and Python-style timestamp formatting.

use std::sync::{Mutex, MutexGuard, PoisonError};

use chrono::{NaiveDateTime, Timelike};

/// Locks `mutex`, recovering the guard when a previous holder panicked (the protected data
/// stays usable: every writer leaves it consistent between statements).
pub fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `value` as `i64`, saturating at `i64::MAX` (graph and JSON integers are `i64`).
#[must_use]
pub fn saturating_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Python `datetime.isoformat()` of a naive timestamp: microseconds, omitted when zero.
/// Sub-microsecond precision truncates; leap-second nanoseconds (>= 1e9) are clamped into the
/// last microsecond of the second.
#[must_use]
pub fn python_isoformat(timestamp: &NaiveDateTime) -> String {
    let micros = (timestamp.nanosecond() / 1_000).min(999_999);
    let base = timestamp.format("%Y-%m-%dT%H:%M:%S");
    if micros == 0 {
        base.to_string()
    } else {
        format!("{base}.{micros:06}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn lock_recovers_from_poison() {
        let mutex = std::sync::Arc::new(Mutex::new(1));
        let poisoner = std::sync::Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("first lock");
            panic!("poison the mutex");
        })
        .join();
        assert!(mutex.is_poisoned());
        *lock(&mutex) += 1;
        assert_eq!(*lock(&mutex), 2);
    }

    fn at(h: u32, m: u32, s: u32, nanos: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 3, 5)
            .and_then(|d| d.and_hms_nano_opt(h, m, s, nanos))
            .expect("valid timestamp")
    }

    #[test]
    fn isoformat_matches_python() {
        assert_eq!(python_isoformat(&at(7, 8, 9, 0)), "2024-03-05T07:08:09");
        assert_eq!(python_isoformat(&at(7, 8, 9, 1_000)), "2024-03-05T07:08:09.000001");
        assert_eq!(python_isoformat(&at(23, 59, 59, 123_456_789)), "2024-03-05T23:59:59.123456");
        assert_eq!(python_isoformat(&at(0, 0, 0, 999)), "2024-03-05T00:00:00");
        // A leap second keeps six fractional digits.
        assert_eq!(python_isoformat(&at(23, 59, 59, 1_999_999_999)), "2024-03-05T23:59:60.999999");
    }

    #[test]
    fn saturating_i64_clamps() {
        assert_eq!(saturating_i64(7), 7);
        assert_eq!(saturating_i64(usize::MAX), i64::MAX);
    }
}
