use num_traits::{Num,pow};
use std::fmt;
// Float vs Int: Num includes Integers. Many Deep Learning ops (Sigmoid, Sqrt, Tanh) only work on Floats (f32/f64).
// You might need to constrain some impl blocks to Float or Real traits.
//
//
// Implement matmul (Matrix Multiplication).
//
// Implement sum (Reduction along an axis).
//
// Implement exp and log (needed for Softmax).
//
// Build a simple Linear layer using the above.
//
#[derive(Debug, Clone)]
struct Tensor<T: Num + Copy> {
    data: Vec<T>,
    // shape is Y,X
    shape: Vec<usize>,
    strides: Vec<usize>,
}
#[derive(Debug)]
pub struct TensorIntoIterator<T: Num + Copy> {
    tensor: Tensor<T>,
    index: usize,
    end: usize,
    window: usize,
}
#[derive(Debug)]
enum TensorError {
    DimensionMismatch { expected: usize, actual: usize },
    EmptyTensor,
    ComputationFailed(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorError::DimensionMismatch { expected, actual } => {
                write!(f, "Wrong Dimensions expected:{},actual:{}.", expected, actual)
            }
            TensorError::EmptyTensor => {
                write!(f, "The passed Tensor is empty.")
            }
            TensorError::ComputationFailed(msg) => write!(f, "Computation Error,{}", msg),
        }
    }
}

impl<T: Num + Copy> Tensor<T> {
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides: Vec<usize> = calculate_strides(&shape);
        Tensor { data, shape, strides }
    }
        
    fn exp_(&mut self)->Result<T,TensorError>{
        for num in self.data.iter_mut(){
            *num = ;
        }
    }
    fn get_index(&self, indices: &[usize]) -> usize {
        let mut index: usize = 0;
        for (i, &dim_index) in indices.iter().enumerate() {
            index += dim_index * self.strides[i];
        }
        index
    }

    fn mul_(&mut self, t2: &Self) -> Result<(), TensorError> {
        if self.shape != t2.shape {
            return Err(TensorError::DimensionMismatch {
                expected: self.shape.len(),
                actual: t2.shape.len(),
            });
        }
        for (value1, value2) in self.data.iter_mut().zip(t2.data.iter()) {
            *value1 = *value1 * *value2;
        }
        Ok(())
    }

    fn mul(&self, t1: Self, t2: Self) -> Result<Self, TensorError> {
        todo!()
    }
    fn is_empty(&self) -> bool {
        todo!()
    }
    // Assuming you have T: Num + Copy + ...
    fn mat_mul(&self, m2: &Self) -> Result<Self, TensorError> {
        // 1. Check Dimensions
        if self.shape.is_empty() || m2.shape.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        // Matrix Multiplication requires: [M x K] * [K x N] = [M x N]
        let rows_self = self.shape[0];
        let cols_self = self.shape[1]; // This is 'K'
        let rows_m2 = m2.shape[0]; // This is also 'K'

        if cols_self != rows_m2 {
            return Err(TensorError::DimensionMismatch {
                expected: cols_self,
                actual: rows_m2,
            });
        }

        // 2. Transpose m2 (to allow row-row dot products)
        // Note: This relies on your transpose implementation returning a new Tensor
        let m2_transposed = 
            transpose(m2)
            .map_err(|e| TensorError::ComputationFailed(format!("Transpose failed: {}", e)))?;

        // 3. Prepare result buffer
        // The result shape will be [rows_self, cols_m2]
        // cols_m2 is effectively rows_m2_transposed after transposition
        let cols_m2 = m2.shape[1];
        let mut result_data = Vec::with_capacity(rows_self * cols_m2); // with capacity is
                                                                       // based and will
                                                                       // let us use less
                                                                       // memory than
                                                                       // necessary while
                                                                       // avoiding reallocs

        // 4. Perform Multiplication using chunks()
        // 'self.strides[0]' is the length of one row (the number of columns)
        let self_row_len = self.strides[0];
        let m2_t_row_len = m2_transposed.strides[0];

        // Iterate over rows of A
        for row_a in self.data.chunks(self_row_len) {
            // Iterate over rows of B_Transposed (which are columns of B)
            for col_b in m2_transposed.data.chunks(m2_t_row_len) {
                // Your dot_product function
                let val = dot_product(row_a, col_b);
                result_data.push(val);
            }
        }

        // 5. Construct final Tensor
        // Result shape is [rows_A, cols_B]
        let new_shape = vec![rows_self, cols_m2];
        Ok(Tensor::new(result_data, new_shape))
    }

    fn mat_from_iter(m: &[Vec<T>], shape: &Vec<usize>) -> Result<Self, TensorError> {
        if shape.len() != 2 {
            return Err(TensorError::ComputationFailed("Shape must be 2D".to_string()));
        }

        let total_size = shape[0] * shape[1];
        let mut data = Vec::with_capacity(total_size);

        for vec in m {
            for num in vec {
                data.push(*num);
            }
        }

        let strides = calculate_strides(shape);
        Ok(Tensor {
            data,
            shape: shape.clone(),
            strides,
        })
    }
    // Real in memory representation:
    // A:0,B:1,C:2,D:3,E:4,F:5,G:6,H:7,I:8,L:9,M:10,N:11
    ///Input For a shape [3, 4], strides are [4, 1]:
    ///
    /// To get to G in a tensor such with a shape(The abstract concept) we need to translate this
    /// into memory all we need to to is to use this formula real_position in memory x,y where x in [0,shape[1]] and  y in [0,shape[0]]
    /// real_position in memory = y*stides[0]+x*stides[1]
    /// when we transpose all we need to to is to rotate everything so this costs O(1) memory wise,
    /// and O(N) time wise where N is the amount of dimensions we are working with. so with a
    /// dimension of
    // [0] = [A,B,C,D]
    // [1] = [E,F,G,H]
    // [2] = [I,L,M,N]
    //
    //To get to the next item we n
    // Output For a shape [4, 3], strides are [1, 4]:
    // [0] = [A,E,I]
    // [1] = [B,F,L]
    // [2] = [C,G,M]
    // [3] = [D,H,N]
    //
    fn transpose_(&mut self) -> Result<(), TensorError> {
        self.shape.reverse();

        self.strides.reverse();

        Ok(())
    }

    fn transpose(input &Self) -> Result<Self, TensorError> {
        let mut t_res = input.clone();
        t_res.transpose_().map_err(|err_msg| err_msg);
        Ok(t_res)
    }

    fn dot_product(v1: &[T], v2: &[T]) -> Result<T, TensorError> {
        if v1.is_empty() || v2.is_empty() {
            return Err(TensorError::ComputationFailed("The vectors must contain values".to_string()));
        }

        assert_eq!(v1.len(), v2.len(), "Vertices lengths should be equal");

        let mut sum = T::zero();
        for (&value1, &value2) in v1.iter().zip(v2) {
            sum = sum + (value1 * value2);
        }
        Ok(sum)
    }
}

// For a shape [3, 4], strides are [4, 1]
// [0] = [A,B,C,D]
// [1] = [E,F,G,H]
// [2] = [I,L,M,N]
// In memory this is just one big array, as it is saved all into a single vector, data
// But we are working with matrices which have multiple dimensions, so we need to translate
// everything from this logical structure to the real memory structure so that we can save a lot of
// memory since everything is not saved sporadically but contiguously.
// The first number in the stides vector is just telling us how many places we need to move to get
// to the next row, since we are working with contigous memory we can't just go down 1 y
// so all we need to do is to move 4 places from where we started to the right to get the actual
// value at index Tensor[1][0], let us say we want to get G: A:0,B:1,C:2,D:3,E:4:F:5,G:6...
// so we need to move 4 places to the right plus 2 to the right to get to our G which as an index
// of 6
// So Row: 1 Column : 2, You take Row: 1 * 4 and Column 2 * 1 sum them togheter and get the real
// position in memory, this is clearly scalable in memory when talking about multiple dims.
// For a shape [2, 3, 4], strides are [12, 4, 1]
fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
fn dot_product<T: Num + Copy>(v1: &[T], v2: &[T]) -> T {
    if v1.is_empty() || v2.is_empty() {
        return T::zero();
    }

    assert_eq!(v1.len(), v2.len(), "Vertices lengths should be equal");

    let mut sum = T::zero();
    for (&value1, &value2) in v1.iter().zip(v2) {
        sum = sum + (value1 * value2);
    }

    sum
}
