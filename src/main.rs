include!("matrix_math.rs");
fn main() {
    let m1 = vec![vec![1, 2], vec![3, 4]];
    let v2 = vec![2, 2];
    match multiplication(&m1, &m1) {
        Ok(matrix) => {
            println!("m1:{:?}\tv2:{:?}\nInt Matrix Mult: {:?}", m1, m1, matrix);
        }
        Err(err_msg) => {
            eprintln!("The matrix multiplication failed: {}", err_msg);
        }
    }

    let v1 = vec![1, 2];
    println!(
        "v1:{:?}\tv2:{:?}\nInt Dot Product: {:?}",
        v1,
        v2,
        dot_product(&v1, &v2)
    );
    println!(
        "og_matrix:{:?}\n transposed matrix : {:?}",
        m1,
        transpose(&m1)
    );
}
