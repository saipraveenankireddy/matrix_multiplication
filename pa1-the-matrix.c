
// CS 683 (Autumn 2023)
// PA 1: The Matrix

// includes
#include <stdio.h>
#include <time.h>			// for time-keeping
#include <xmmintrin.h> 
#include <immintrin.h>
		                       // for intrinsic functions

// defines
// NOTE: you can change this value as per your requirement
#define BLOCK_SIZE	20		// size of the block
/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
}

/**
 * @brief 		Initialize result matrix of given dimension with 0.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_result_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = 0.0;
		}
	}
}

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 */
void normal_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
}

/**
 * @brief 		Task 1: Performs matrix multiplication of two matrices using blocking.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the block size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size) {


		for(int i=0; i< dim ;i+=block_size){
			for(int j =0; j< dim ; j+=block_size) {
				for(int k=0; k<dim ; k+= block_size) {
					for(int i1 = i; i1 < i+block_size ; i1++){
						for(int j1=j; j1<j+block_size ; j1++) {
							for(int k1 =k;k1<k+block_size ; k1++) {
								C[i1*dim+j1] += A[i1*dim+k1] * B[k1*dim+j1] ;
							}
						}
					}
				}
			}
		}
}

/**
 * @brief 		Task 2: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int dim) {
double B_transposed[dim * dim ];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            B_transposed[j * dim + i] = B[i * dim + j];
        }
    }
		    for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
			    __m256d sum = _mm256_setzero_pd();
			    for (int k = 0; k < dim; k += 4) {
				__m256d a = _mm256_loadu_pd(&A[i * dim + k]);
				__m256d b = _mm256_loadu_pd(&B_transposed[j * dim + k]);
				sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
			    }
			    double result[4] ; // Initialize result array
			    _mm256_storeu_pd(result, sum);
			    C[i * dim + j] = result[0] + result[1] + result[2] + result[3];
			}
		    }
}

/**
 * @brief 		Task 3: Performs matrix multiplication of two matrices using software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void prefetch_mat_mul(double *A, double *B, double *C, int dim) {
   
for(int i =0;i<dim;i++){
	for(int j=0;j<dim;j++) {
	    C[i*dim+j] =0;
	    		for(int k=0;k<dim;k+=4){
			_mm_prefetch(&A[i*dim+k+4],_MM_HINT_T0);
			_mm_prefetch(&A[i*dim+k+5],_MM_HINT_T0);
			_mm_prefetch(&A[i*dim+k+6],_MM_HINT_T0);
			_mm_prefetch(&A[i*dim+k+7],_MM_HINT_T0);
			_mm_prefetch(&B[(k+4)* dim +j],_MM_HINT_T0);
			_mm_prefetch(&B[(k+5) *dim +j],_MM_HINT_T0);
			_mm_prefetch(&B[(k+6) *dim +j],_MM_HINT_T0);
			_mm_prefetch(&B[(k+7) *dim +j],_MM_HINT_T0);
			C[i * dim + j] += (A[i * dim + k] * B[k * dim + j]) + (A[i * dim + k+1] * B[(k+1)*dim + j]) + (A[i * dim + k+2] * B[(k+2) * dim + j]) + (A[i * dim + k+3] * B[(k+3)*dim + j]) ;
			
				
	}
	
}
}
}

/**
 * @brief 		Bonus Task 1: Performs matrix multiplication of two matrices using blocking along with SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

}

/**
 * @brief 		Bonus Task 2: Performs matrix multiplication of two matrices using blocking along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

}

/**
 * @brief 		Bonus Task 3: Performs matrix multiplication of two matrices using SIMD instructions along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_prefetch_mat_mul(double *A, double *B, double *C, int dim) {

}

/**
 * @brief 		Bonus Task 4: Performs matrix multiplication of two matrices using blocking along with SIMD instructions and software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void blocking_simd_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {
#define dim 800;

	if ( argc <= 1 ) {
		printf("Pass the matrix dimension as argument :)\n\n");
		return 0;
	}

	else {
		int matrix_dim = atoi(argv[1]);

		// variables definition and initialization
		clock_t t_normal_mult, t_blocking_mult, t_prefetch_mult, t_simd_mult, t_blocking_simd_mult, t_blocking_prefetch_mult, t_simd_prefetch_mult, t_blocking_simd_prefetch_mult;
		double time_normal_mult, time_blocking_mult, time_prefetch_mult, time_simd_mult, time_blocking_simd_mult, time_blocking_prefetch_mult, time_simd_prefetch_mult, time_blocking_simd_prefetch_mult;

		double *A = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *B = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, matrix_dim, matrix_dim);
		initialize_matrix(B, matrix_dim, matrix_dim);

		// perform normal matrix multiplication
		t_normal_mult = clock();
		normal_mat_mul(A, B, C, matrix_dim);
		t_normal_mult = clock() - t_normal_mult;

		time_normal_mult = ((double)t_normal_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Normal matrix multiplication took %f seconds to execute \n\n", time_normal_mult);
	
	
				
	#ifdef OPTIMIZE_BLOCKING
		// Task 1: perform blocking matrix multiplication

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_mult = clock();
		blocking_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_mult = clock() - t_blocking_mult;

		time_blocking_mult = ((double)t_blocking_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking matrix multiplication took %f seconds to execute \n", time_blocking_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_mult);
		printf("\n");
	
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 2: perform matrix multiplication with SIMD instructions

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_simd_mult = clock();
		simd_mat_mul(A, B, C, matrix_dim);
		t_simd_mult = clock() - t_simd_mult;

		time_simd_mult = ((double)t_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD matrix multiplication took %f seconds to execute \n", time_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_mult);
	
	#endif

	#ifdef OPTIMIZE_PREFETCH
		// Task 3: perform matrix multiplication with prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_prefetch_mult = clock();
		prefetch_mat_mul(A, B, C, matrix_dim);
		t_prefetch_mult = clock() - t_prefetch_mult;

		time_prefetch_mult = ((double)t_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Prefetching matrix multiplication took %f seconds to execute \n", time_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_prefetch_mult);
	
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD
		// Bonus Task 1: perform matrix multiplication using blocking along with SIMD instructions

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_simd_mult = clock();
		blocking_simd_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_mult = clock() - t_blocking_simd_mult;

		time_blocking_simd_mult = ((double)t_blocking_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD matrix multiplication took %f seconds to execute \n", time_blocking_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_mult);
	#endif

	#ifdef OPTIMIZE_BLOCKING_PREFETCH
		// Bonus Task 2: perform matrix multiplication using blocking along with software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_prefetch_mult = clock();
		blocking_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_prefetch_mult = clock() - t_blocking_prefetch_mult;

		time_blocking_prefetch_mult = ((double)t_blocking_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with prefetching matrix multiplication took %f seconds to execute \n", time_blocking_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_prefetch_mult);
	#endif

	#ifdef OPTIMIZE_SIMD_PREFETCH
		// Bonus Task 3: perform matrix multiplication using SIMD instructions along with software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_simd_prefetch_mult = clock();
		simd_prefetch_mat_mul(A, B, C, matrix_dim);
		t_simd_prefetch_mult = clock() - t_simd_prefetch_mult;

		time_simd_prefetch_mult = ((double)t_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD with prefetching matrix multiplication took %f seconds to execute \n", time_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_prefetch_mult);
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD_PREFETCH
		// Bonus Task 4: perform matrix multiplication using blocking, SIMD instructions and software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_simd_prefetch_mult = clock();
		blocking_simd_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_prefetch_mult = clock() - t_blocking_simd_prefetch_mult;

		time_blocking_simd_prefetch_mult = ((double)t_blocking_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD and prefetching matrix multiplication took %f seconds to execute \n", time_blocking_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_prefetch_mult);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
