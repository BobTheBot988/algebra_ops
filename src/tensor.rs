use num_traits::Num;
#[derive(Debug, Clone)]
struct Tensor<T: Num + Copy> {
    data: Vec<T>,
    // shape is Y,X
    shape: Vec<usize>,
    strides: Vec<usize>,
}
impl<T: Num + Copy> Tensor<T> {
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides: Vec<usize> = calculate_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
        }
    }
    fn get_index(&self, indices: &[usize]) -> usize {
        let mut index: usize = 0;
        for (i, &dim_index) in indices.iter().enumerate() {
            index += dim_index * self.strides[i];
        }
        index
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
// position in memory, this is clearly scalable in memory.
// For a shape [2, 3, 4], strides are [12, 4, 1]
fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i * 1];
    }
    strides
}
