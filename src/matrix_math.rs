use num_traits::Num;
#[derive(Debug)]
enum MatrixError {
    DimensionMismatch { expected: usize, actual: usize },
    EmptyMatrix,
}

fn transpose<T: Num + Copy + Default>(v: &[Vec<T>]) -> Result<Vec<Vec<T>>, MatrixError> {
    if v.is_empty() {
        return Err(MatrixError::EmptyMatrix);
    }
    let mut result_vec = Vec::with_capacity(v.len());

    for x in 0..v[0].len() {
        let mut tmp_vec = Vec::with_capacity(v[0].len());
        (0..v.len()).for_each(|y| {
            tmp_vec.push(v[y][x]);
        });
        result_vec.push(tmp_vec);
    }
    Ok(result_vec)
}

fn dot_product<T: Num + Copy + Default>(v1: &[T], v2: &[T]) -> T {
    if v1.is_empty() {
        return T::default();
    }

    assert_eq!(v1.len(), v2.len(), "Vertices lengths should be equal");

    let mut sum = T::default();
    for (&value1, &value2) in v1.iter().zip(v2) {
        sum = sum + (value1 * value2);
    }

    sum
}

// 1. Matrix Multiplication (M x K) * (K x N) -> (M x N)
fn multiplication<T: Num + Copy + Default>(
    m1: &[Vec<T>],
    m2: &[Vec<T>],
) -> Result<Vec<Vec<T>>, MatrixError> {
    if m1.is_empty() || m2.is_empty() {
        return Err(MatrixError::EmptyMatrix);
    }
    let cols_m1 = m1[0].len();
    let rows_m2 = m2.len();
    if cols_m1 != rows_m2 {
        return Err(MatrixError::DimensionMismatch {
            expected: cols_m1,
            actual: rows_m2,
        });
    }

    let mut result_matrix = Vec::with_capacity(m1.len());
    let res = transpose(m2);

    let m2_transposed = match res {
        Ok(matrix) => matrix,
        Err(err_msg) => {
            return Err(format!("Transposition failed: {:?}", err_msg));
        }
    };

    for row_a in m1 {
        let mut row_result = Vec::with_capacity(m2_transposed.len());
        for row_b in &m2_transposed {
            let val = dot_product(row_a, row_b);
            row_result.push(val);
        }

        result_matrix.push(row_result);
    }
    Ok(result_matrix)
}
