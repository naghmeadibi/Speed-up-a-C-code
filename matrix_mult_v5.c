#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void fill(double* x, int n) {

    int i;
    for (i=0, n=n*n; i<n; i++, x++)
        *x = ((double) (1 + rand() % 12345)) / ((double) (1 + rand() % 6789));
}

void matrix_mult_index (int n, double* a, double* b, double* c) {
  int i, j, k;
  for (i=0; i<n; i++)
    for (j=0; j<n; j++) {
      c[i*n+j] =0;
      for(k = 0; k < n; k++)
        c[i*n+j] += a[i*n+k] * b[k*n+j];
    }
}

void matrix_mult_ptr_reg (int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k;
    for (i=0; i<n; i++, a+=n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for(k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt+=n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_ptr_no_reg (int n, double* a, double* b, double* c) {
    double cij;
    double *at, *bt;
    int i, j, k;
    for (i=0; i<n; i++, a+=n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for(k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt+=n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_transpose_vector(int n, double* a, double* b, double* c){
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
    int i, j, k;

    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }

    register double *btt, *BT;

    // Transpose matrix B into bT
    for (i = 0, BT = b; i < n; i++) {
        for (j = 0, btt = &bT[i]; j < n; j++, btt += n, BT++) {
            *btt = *BT;
        }
    }

    typedef double v4df __attribute__ ((vector_size (32)));

    v4df ctij;
    register v4df cij;
    register v4df *at, *bt;




    for(i = 0; i < n; i++, a += n) {
        for(j = 0, bt = (v4df *) bT; j < n; j++, c++) {
            cij = (v4df) {0, 0, 0, 0};
            for(k = 0, at = (v4df *) a; k < n; k += 4, at++, bt++) {
                cij += *at * *bt;
            }
            ctij = cij;
            *c = ctij[0] + ctij[1] + ctij[2] + ctij[3];
        }
    }





    _aligned_free(bT);
}


void matrix_mult_transpose_vector_unroll(int n, double* a, double* b, double* c) {
    typedef double v4df __attribute__ ((vector_size(32)));
    v4df ctij;
    register double *btt, *BT;
    register v4df cij, cij1, cij2, cij3;
    register v4df *at, *bt;
    register int i, j, k;
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }



    // Transpose matrix B into bT
    for (i = 0, BT = b; i < n; i++) {
        for (j = 0, btt = &bT[i]; j < n; j++, btt += n, BT++) {
            *btt = *BT;
        }
    }


    // Multiplication
    for (i = 0; i < n; i++, a += n) {
        for (j = 0, bt = (v4df*) bT; j < n; j++, c++) {
            cij = (v4df) {0, 0, 0, 0};
            cij1 = (v4df) {0, 0, 0, 0};
            cij2 = (v4df) {0, 0, 0, 0};
            cij3 = (v4df) {0, 0, 0, 0};
            for (k = 0, at = (v4df*) a; k < n; k+=16, at+=4, bt+=4) {
                cij += *at * *bt;
                cij1 += *(at+1) * *(bt+1);
                cij2 += *(at+2) * *(bt+2);
                cij3 += *(at+3) * *(bt+3);
            }
            ctij = (cij + cij1) + (cij2 + cij3);
            *c = (ctij[0] + ctij[1]) + (ctij[2] + ctij[3]);
        }
    }
    _aligned_free(bT);
}

void matrix_mult_transpose(int n, double* a, double* b, double* c){
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }

    register double cij;
    register double *at, *bt, *btt, *BT;
    register int i, j, k;

    // Transpose matrix B into bT
    for (i = 0, BT = b; i < n; i++) {
        for (j = 0, btt = &bT[i]; j < n; j++, btt += n, BT++) {
            *btt = *BT;
        }
    }

    // Perform multiplication with transposed matrix bT
    for (i=0; i<n; i++, a+=n)
        for (j = 0, bt = bT; j < n; j++, c++) {
            cij = 0;
            for(k = 0, at = a; k < n; k++, at++, bt++)
                cij += *at * *bt;
            *c = cij;
        }
    _aligned_free(bT);
}

void matrix_mult_transpose_unrolling(int n, double* a, double* b, double* c){
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }

    register double cij0, cij1, cij2, cij3;
    register double *at0, *bt0, *at1, *bt1, *at2, *bt2, *at3, *bt3, *btt0, *BT0, *btt1, *BT1, *btt2, *BT2, *btt3, *BT3;
    register int i, j, k;

    // Transpose matrix B into bT
    for (i = 0, BT0 = b, BT1 = b+1, BT2 = b+2, BT3 = b+3; i < n; i++) {
        for (j = 0, btt0 = &bT[i], btt1 = &bT[i]+n, btt2 = &bT[i]+2*n, btt3 = &bT[i]+3*n;
                j < n/4; j++, btt0 += 4*n, btt1 += 4*n, btt2 += 4*n, btt3 += 4*n, BT0+=4, BT1+=4, BT2+=4, BT3+=4) {
            *btt0 = *BT0;
            *btt1 = *BT1;
            *btt2 = *BT2;
            *btt3 = *BT3;
        }
    }

    // Perform multiplication with transposed matrix bT
    for (i=0; i<n; i++, a+=n)
        for (j = 0, bt0 = bT, bt1 = bT+1, bt2 = bT+2, bt3 = bT+3; j < n; j++, c++) {
            cij0 = 0;
            cij1 = 0;
            cij2 = 0;
            cij3 = 0;
            for(k = 0, at0 = a, at1 = a+1, at2 = a+2, at3 = a+3;
                k < n/4; k++, at0+=4, bt0+=4, at1+=4, bt1+=4, at2+=4, bt2+=4, at3+=4, bt3+=4) {
                cij0 += *at0 * *bt0;
                cij1 += *at1 * *bt1;
                cij2 += *at2 * *bt2;
                cij3 += *at3 * *bt3;
            }
            *c = (cij0 + cij1) + (cij2 + cij3);
        }
    _aligned_free(bT);
}

void matrix_mult_block(int n, int block_size, double* a, double* b, double* c) {
    // Initialize matrix C to zero
    for (int i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }

    register double *a_block_start, *b_block_start, *c_block_start;
    register double *a_ptr, *b_ptr, *c_ptr;
    register int ii, jj, kk, i, j, k;
    register double sum;

    // Block multiplication
    for (ii = 0; ii < n; ii += block_size, a += (n*block_size), c += (n*block_size)) {
        for (jj = 0, c_block_start = c; jj < n; jj += block_size, c_block_start += block_size) {
            for (kk = 0, a_block_start = a, b_block_start = (b+jj); kk < n; kk += block_size, a_block_start += block_size, b_block_start += (block_size*n)) {
                // Multiply blocks
                for (i = 0, c_ptr = c_block_start, a_ptr = a_block_start; i < block_size; ++i, c_ptr += (n-block_size), a_ptr += n) {
                    for (j = 0, b_ptr = b_block_start; j < block_size; ++j, ++c_ptr, ++b_ptr) {
                        sum = *c_ptr;
                        for (int k = 0; k < block_size; ++k) {
                            sum += a_ptr[k] * b_ptr[k * n];
                        }
                        *c_ptr = sum;
                    }
                }
            }
        }
    }
}



void matrix_mult_transpose_vector_OMP(int n, double* a, double* b, double* c){
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);


    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }

    register double *btt, *BT;
    register int ii, jj;
    // Transpose matrix B into bT
    for (ii = 0, BT = b; ii < n; ii++) {
        for (jj = 0, btt = &bT[ii]; jj < n; jj++, btt += n, BT++) {
            *btt = *BT;
        }
    }
    #pragma omp parallel num_threads(16)
    {
        typedef double v4df __attribute__ ((vector_size (32)));
        int ID = omp_get_thread_num();
        int i, j, k;
        v4df ctij;
        v4df cij;
        v4df *at, *bt;
        double* aa = a+((n*n/16)*ID);
        double* cc = c+((n*n/16)*ID);
        //printf("%d \n", cc-c);

        for(i = ID*(n/16); i < ((ID+1)*(n/16)); i++, aa += n) {
            for(j = 0, bt = (v4df *) bT; j < n; j++, cc++) {
                cij = (v4df) {0, 0, 0, 0};
                for(k = 0, at = (v4df *) aa; k < n; k += 4, at++, bt++) {
                    cij += *at * *bt;
                }
                ctij = cij;
                *cc = ctij[0] + ctij[1] + ctij[2] + ctij[3];
                //printf("%d \n", cc-c);
            }
        }
    }





    _aligned_free(bT);
}


void matrix_mult_transpose_vector_unroll_OMP(int n, double* a, double* b, double* c) {
    typedef double v4df __attribute__ ((vector_size(32)));
    register double *btt, *BT;
    register int ii, jj;
    double *bT  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
    // Check if memory allocation was successful
    if (bT == NULL) {
        printf("Memory Allocation Error for Transposed Matrix\n\n");
        return;
    }
    // Transpose matrix B into bT
    for (ii = 0, BT = b; ii < n; ii++) {
        for (jj = 0, btt = &bT[ii]; jj < n; jj++, btt += n, BT++) {
            *btt = *BT;
        }
    }
    // Multiplication
    #pragma omp parallel num_threads(16)
    {
        int ID = omp_get_thread_num();
        v4df cij, cij1, cij2, cij3;
        v4df *at, *bt;
        v4df ctij;
        int i, j, k;
        double* aa = a+((n*n/16)*ID);
        double* cc = c+((n*n/16)*ID);

        for (i = ID*(n/16); i < (ID+1)*(n/16); i++, aa += n) {
            for (j = 0, bt = (v4df*) bT; j < n; j++, cc++) {
                cij = (v4df) {0, 0, 0, 0};
                cij1 = (v4df) {0, 0, 0, 0};
                cij2 = (v4df) {0, 0, 0, 0};
                cij3 = (v4df) {0, 0, 0, 0};
                for (k = 0, at = (v4df*) aa; k < n; k+=16, at+=4, bt+=4) {
                    cij += *at * *bt;
                    cij1 += *(at+1) * *(bt+1);
                    cij2 += *(at+2) * *(bt+2);
                    cij3 += *(at+3) * *(bt+3);
                }
                ctij = (cij + cij1) + (cij2 + cij3);
                *cc = (ctij[0] + ctij[1]) + (ctij[2] + ctij[3]);
            }
        }

    }

    _aligned_free(bT);
}


int main()
{
    clock_t t0, t1;
    int n, ref;

    do{
        printf("Input size of matrix, n = ");
        scanf("%d", &n);

        ref = 0;

        double *A  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/); //  64 is cache line size
        double *B  = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double *C1 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);
        double *C2 = (double*)_aligned_malloc(n * n * sizeof(double), 64 /*sizeof(double)*/);

        if(A == NULL || B == NULL || C1 == NULL || C2 == NULL){
            printf("Memory Allocation Error\n\n");
            return(-1);
        }

        unsigned int seed = time(NULL);
        printf("\nseed = %u\n", seed);

        srand(seed);
        fill(A, n);
        fill(B, n);

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_index (y/n)? ");
        if(getchar() == 'y'){
            ref = 1;
            t0 = clock();
            matrix_mult_index(n, A, B, C1);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_index = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_reg (y/n)? ");
        if(getchar() == 'y'){
            ref++;
            t0 = clock();
            matrix_mult_ptr_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_reg = %0.2f s\n", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_no_reg (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_ptr_no_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_no_reg = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_block (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;

            int block_size;
            printf("\n\tInput size of block = ");
            scanf("%d", &block_size);

            t0 = clock();
            matrix_mult_block(n, block_size, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_block = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_unrolling (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_unrolling(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_unrolling = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector_unroll (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector_unroll(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector_unroll = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector_OMP (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector_OMP(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector_OMP = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector_unroll_OMP (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector_unroll_OMP(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector_unroll_OMP = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }



        printf("\n\n\nEnd Of Execution\n\n");

        if(ref == 2){
            int i;
            double *c1, *c2;
            printf("\n\nStart of Compare: ");
            for(i=0, c1=C1, c2=C2, n=n*n; i<n; i++, c1++, c2++){
//              if(*c1 != *c2)
                if(abs((*c1 - *c2) / *c1) > 1E-10)
                    break;
                if(i % (n/20) == 0)
                    printf(".");
            }

            if(i != n)
                printf(" Ooops, Error Found @ %d: %0.3f vs %0.3f\n\n",i, *c1, *c2);
            else
                printf(" OK, OK, Matrixes are equivalent.\n\n");
        }
        else
            printf("\n\nNo Compare due to No Reference or No Data.\n\n");

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C1);
        _aligned_free(C2);

        fflush(stdin);
        printf("\n\nDo you want to continue (y/n)? ");

    } while(getchar() == 'y');

    return 0;
}
