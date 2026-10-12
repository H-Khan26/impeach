//! Small helpers shared by every crate: poison-tolerant locking and saturating integer
//! conversion.

use std::sync::{Mutex, MutexGuard, PoisonError};

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn saturating_i64_clamps() {
        assert_eq!(saturating_i64(7), 7);
        assert_eq!(saturating_i64(usize::MAX), i64::MAX);
    }
}
