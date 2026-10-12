//! One-to-one assignment of key rows to detected pairs.
//!
//! [`linear_sum_assignment`] is a port of SciPy's `scipy.optimize.linear_sum_assignment`
//! (`rectangular_lsap.cpp`: the shortest augmenting path algorithm of Crouse, "On implementing
//! 2D rectangular assignment algorithms", 2016). It follows SciPy step by step: rows are
//! augmented in order, the remaining columns are scanned in SciPy's order (initially from the
//! last column to the first), and on equal reduced cost a still-unassigned column replaces the
//! current pick. So it picks the same optimum as the Python evaluator, including among tied
//! optima (a constant matrix gives the identity), up to floating-point summation order.

use serde::{Deserialize, Serialize};

use crate::error::{EvalError, Result};

/// How key rows and detected pairs are paired.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentMethod {
    /// Optimal assignment maximising the total admissible score.
    #[default]
    Optimal,
    /// Greedy highest-score-first (the Python fallback when SciPy is missing).
    Greedy,
}

impl AssignmentMethod {
    /// The report's `assignment` string, as the Python evaluator writes it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "optimal (scipy linear_sum_assignment)",
            Self::Greedy => "greedy (scipy not installed)",
        }
    }
}

/// Solves the rectangular linear sum assignment problem on a row-major `cost` matrix with
/// `rows` x `cols` entries, returning `(row, col)` pairs sorted by row. Every row (or every
/// column, for a tall matrix) is assigned.
///
/// # Errors
/// [`EvalError::InvalidScores`] when the matrix holds NaN or (after negation for
/// `maximize`) negative infinity, as SciPy raises `ValueError`.
pub fn linear_sum_assignment(
    cost: &[f64],
    rows: usize,
    cols: usize,
    maximize: bool,
) -> Result<Vec<(usize, usize)>> {
    debug_assert_eq!(cost.len(), rows * cols, "cost matrix shape");
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let transpose = cols < rows;
    let (nr, nc) = if transpose { (cols, rows) } else { (rows, cols) };
    let mut c = vec![0.0; nr * nc];
    for i in 0..rows {
        for j in 0..cols {
            let value = if maximize { -cost[i * cols + j] } else { cost[i * cols + j] };
            if transpose {
                c[j * rows + i] = value;
            } else {
                c[i * cols + j] = value;
            }
        }
    }
    if c.iter().any(|x| x.is_nan() || *x == f64::NEG_INFINITY) {
        return Err(EvalError::InvalidScores);
    }

    let mut u = vec![0.0; nr];
    let mut v = vec![0.0; nc];
    let mut shortest = vec![f64::INFINITY; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row: Vec<Option<usize>> = vec![None; nr];
    let mut row4col: Vec<Option<usize>> = vec![None; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0; nc];

    for cur_row in 0..nr {
        // Shortest augmenting path from `cur_row`.
        let mut min_val = 0.0;
        let mut num_remaining = nc;
        for (it, slot) in remaining.iter_mut().enumerate() {
            *slot = nc - it - 1;
        }
        sr.fill(false);
        sc.fill(false);
        shortest.fill(f64::INFINITY);

        let mut i = cur_row;
        let sink = loop {
            let mut index = None;
            let mut lowest = f64::INFINITY;
            sr[i] = true;
            for (it, &j) in remaining[..num_remaining].iter().enumerate() {
                let r = min_val + c[i * nc + j] - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j].is_none()) {
                    lowest = shortest[j];
                    index = Some(it);
                }
            }
            min_val = lowest;
            let Some(index) = index.filter(|_| min_val != f64::INFINITY) else {
                return Err(EvalError::InvalidScores);
            };
            let j = remaining[index];
            sc[j] = true;
            num_remaining -= 1;
            remaining[index] = remaining[num_remaining];
            match row4col[j] {
                None => break j,
                Some(row) => i = row,
            }
        };

        // Dual update.
        u[cur_row] += min_val;
        for row in 0..nr {
            if sr[row] && row != cur_row {
                let col = col4row[row].expect("rows on the path are assigned");
                u[row] += min_val - shortest[col];
            }
        }
        for col in 0..nc {
            if sc[col] {
                v[col] -= min_val - shortest[col];
            }
        }

        // Augment along the path.
        let mut j = sink;
        loop {
            let row = path[j];
            row4col[j] = Some(row);
            let previous = col4row[row].replace(j);
            if row == cur_row {
                break;
            }
            j = previous.expect("rows on the path other than the start are assigned");
        }
    }

    let mut pairs: Vec<(usize, usize)> = col4row
        .iter()
        .enumerate()
        .map(|(row, col)| {
            let col = col.expect("every row of the wide matrix is assigned");
            if transpose {
                (col, row)
            } else {
                (row, col)
            }
        })
        .collect();
    pairs.sort_unstable();
    Ok(pairs)
}

/// Greedy highest-score-first matching over positive admissible scores; ties go to the
/// earlier entry in row-major order.
fn greedy(admissible: &[f64], cols: usize) -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = (0..admissible.len()).collect();
    order.sort_by(|a, b| admissible[*b].total_cmp(&admissible[*a]));
    let mut used_rows = std::collections::HashSet::new();
    let mut used_cols = std::collections::HashSet::new();
    let mut pairs = Vec::new();
    for flat in order {
        if admissible[flat] <= 0.0 || admissible[flat].is_nan() {
            break;
        }
        let (r, c) = (flat / cols, flat % cols);
        if !used_rows.contains(&r) && !used_cols.contains(&c) {
            used_rows.insert(r);
            used_cols.insert(c);
            pairs.push((r, c));
        }
    }
    pairs
}

/// Returns `(key_row, detected_col)` pairs with score >= `threshold` (and > 0), one-to-one,
/// sorted. Scores below the threshold are zeroed before assignment and dropped afterwards.
///
/// # Errors
/// [`EvalError::InvalidScores`] when the optimal solver rejects the matrix.
pub fn assign_one_to_one(
    scores: &[Vec<f64>],
    threshold: f64,
    method: AssignmentMethod,
) -> Result<Vec<(usize, usize)>> {
    let rows = scores.len();
    let cols = scores.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let admissible: Vec<f64> =
        scores.iter().flatten().map(|&s| if s >= threshold { s } else { 0.0 }).collect();
    let pairs = match method {
        AssignmentMethod::Optimal => linear_sum_assignment(&admissible, rows, cols, true)?,
        AssignmentMethod::Greedy => greedy(&admissible, cols),
    };
    let mut kept: Vec<(usize, usize)> = pairs
        .into_iter()
        .filter(|&(r, c)| scores[r][c] >= threshold && scores[r][c] > 0.0)
        .collect();
    kept.sort_unstable();
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lsa(rows: &[&[f64]], maximize: bool) -> Vec<(usize, usize)> {
        let cols = rows[0].len();
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        linear_sum_assignment(&flat, rows.len(), cols, maximize).unwrap()
    }

    #[test]
    fn matches_scipy_examples() {
        // scipy docs example: cost [[4,1,3],[2,0,5],[3,2,2]] -> rows [0,1,2], cols [1,0,2].
        assert_eq!(
            lsa(&[&[4.0, 1.0, 3.0], &[2.0, 0.0, 5.0], &[3.0, 2.0, 2.0]], false),
            [(0, 1), (1, 0), (2, 2)]
        );
        // A constant matrix gives the identity (scipy issue #11602).
        assert_eq!(lsa(&[&[1.0; 3], &[1.0; 3], &[1.0; 3]], false), [(0, 0), (1, 1), (2, 2)]);
        assert_eq!(lsa(&[&[0.0; 3], &[0.0; 3]], true), [(0, 0), (1, 1)]);
    }

    #[test]
    fn rectangular_both_ways() {
        // Wide: every row assigned.
        assert_eq!(lsa(&[&[1.0, 5.0, 2.0], &[4.0, 1.0, 9.0]], true), [(0, 1), (1, 2)]);
        // Tall: every column assigned.
        assert_eq!(lsa(&[&[1.0, 4.0], &[5.0, 1.0], &[2.0, 9.0]], true), [(1, 0), (2, 1)]);
        assert_eq!(lsa(&[&[3.0], &[7.0], &[7.0]], true), [(1, 0)]);
    }

    #[test]
    fn rejects_nan() {
        assert!(linear_sum_assignment(&[f64::NAN, 1.0], 1, 2, true).is_err());
        assert!(linear_sum_assignment(&[f64::INFINITY, 1.0], 1, 2, true).is_err());
    }

    #[test]
    fn optimal_beats_greedy_when_it_should() {
        let scores = vec![vec![0.95, 0.80], vec![0.90, 0.10]];
        let optimal = assign_one_to_one(&scores, 0.75, AssignmentMethod::Optimal).unwrap();
        assert_eq!(optimal, [(0, 1), (1, 0)]);
        let greedy = assign_one_to_one(&scores, 0.75, AssignmentMethod::Greedy).unwrap();
        assert_eq!(greedy, [(0, 0)]);
    }

    #[test]
    fn threshold_filters_pairs() {
        let scores = vec![vec![0.95, 0.10], vec![0.60, 0.20]];
        assert_eq!(assign_one_to_one(&scores, 0.75, AssignmentMethod::Optimal).unwrap(), [(0, 0)]);
        assert!(assign_one_to_one(&[], 0.75, AssignmentMethod::Optimal).unwrap().is_empty());
        assert!(assign_one_to_one(&[vec![]], 0.75, AssignmentMethod::Optimal).unwrap().is_empty());
        // Zero scores never count, even with a non-positive threshold.
        assert!(assign_one_to_one(&[vec![0.0]], 0.0, AssignmentMethod::Optimal)
            .unwrap()
            .is_empty());
    }
}
