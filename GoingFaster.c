#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void fill(double* x, long long int n) {
    for (; n; n--)
        *x++ = ((double) (1 + rand() % 12345)) / ((double) (1 + rand() % 6789));
}

void firf(long long int na, long long int nx, double *a, double *x, double *y){
    for(long long int i = 0; i < (nx-na); i++){
        y[i] = 0;
        for(long long int j = 0; j < na ; j++)
            y[i] += x[i+j] * a[j];
    }
}

void my_fir_reg_point(long long int na, long long int nx, double *a, double *x, double *y){
    register double *at, *xt;
    register long long int i, j;

    for(i = 0; i < (nx-na); i++, y++, x++){
        *y = 0;
        for(j = 0, at = a, xt = x; j < na ; j++, at++, xt++)
            *y += *xt * *at;
    }
}

void my_fir_unrolling(long long int na, long long int nx, double *a, double *x, double *y){
    register double *at, *yt0, *yt1, *yt2, *yt3, *xt0, *xt1, *xt2, *xt3;
    register long long int i, j;

    for(i = 0, yt0 = y, yt1 = y+1, yt2 = y+2, yt3 = y+3; i < (nx-na)/4; i++, yt0+=4, yt1+=4, yt2+=4, yt3+=4, x+=4){
        *yt0 = 0;
        *yt1 = 0;
        *yt2 = 0;
        *yt3 = 0;
        for(j = 0, at = a, xt0 = x, xt1 = x+1, xt2 = x+2, xt3 = x+3; j < na ; j++, at++, xt0++, xt1++, xt2++, xt3++) {
            *yt0 += *xt0 * *at;
            *yt1 += *xt1 * *at;
            *yt2 += *xt2 * *at;
            *yt3 += *xt3 * *at;
        }
    }
}

void my_fir_unrolling_vector(long long int na, long long int nx, double *a, double *x, double *y){
    typedef double v4df __attribute__ ((vector_size(32)));
    register v4df *at, *xt0;
    register v4df yt0, yt1, yt2, yt3;
    register long long int i, j;

    for(i = 0; i < (nx-na)/4; i++, x+=4, y+=4){
        yt0 =(v4df) {0, 0, 0, 0};
        yt1 =(v4df) {0, 0, 0, 0};
        yt2 =(v4df) {0, 0, 0, 0};
        yt3 =(v4df) {0, 0, 0, 0};

        for(j = 0, at =(v4df*) a, xt0 =(v4df*) x; j < na/4 ; j++, at++, xt0++) {
            yt0 += *xt0 * *at;
            yt1 += ((v4df) {*(x+1+j*4), *(x+2+j*4), *(x+3+j*4), *(x+4+j*4)}) * *at;
            yt2 += ((v4df) {*(x+2+j*4), *(x+3+j*4), *(x+4+j*4), *(x+5+j*4)}) * *at;
            yt3 += ((v4df) {*(x+3+j*4), *(x+4+j*4), *(x+5+j*4), *(x+6+j*4)}) * *at;
        }
        *y = yt0[0] + yt0[1] + yt0[2] + yt0[3];
        *(y+1) = yt1[0] + yt1[1] + yt1[2] + yt1[3];
        *(y+2) = yt2[0] + yt2[1] + yt2[2] + yt2[3];
        *(y+3) = yt3[0] + yt3[1] + yt3[2] + yt3[3];
    }
}

void omp(long long int na, long long int nx, double *a, double *x, double *y, int id) {
    typedef double v4df __attribute__ ((vector_size(32)));
        register v4df *at, *xt0;
        register v4df yt0, yt1, yt2, yt3;
        register long long int i, j;
        register double *yt, *xt, *xtt;

        register long long int s = id*(nx-na)/(8);
        xt = x + s;
        yt = y + s;


        for(i = 0; i < (nx-na)/(4*8); i++, xt+=4, yt+=4){
            yt0 =(v4df) {0, 0, 0, 0};
            yt1 =(v4df) {0, 0, 0, 0};
            yt2 =(v4df) {0, 0, 0, 0};
            yt3 =(v4df) {0, 0, 0, 0};

            for(j = 0, at =(v4df*) a, xt0 =(v4df*) xt, xtt = xt; j < na/4 ; j++, at++, xt0++, xtt+=4) {
                yt0 += *xt0 * *at;
                yt1 += ((v4df) {*(xtt+1), *(xtt+2), *(xtt+3), *(xtt+4)}) * *at;
                yt2 += ((v4df) {*(xtt+2), *(xtt+3), *(xtt+4), *(xtt+5)}) * *at;
                yt3 += ((v4df) {*(xtt+3), *(xtt+4), *(xtt+5), *(xtt+6)}) * *at;
            }
            *yt = (yt0[0] + yt0[1]) + (yt0[2] + yt0[3]);
            *(yt+1) = (yt1[0] + yt1[1]) + (yt1[2] + yt1[3]);
            *(yt+2) = (yt2[0] + yt2[1]) + (yt2[2] + yt2[3]);
            *(yt+3) = (yt3[0] + yt3[1]) + (yt3[2] + yt3[3]);
        }
}

void fir(long long int na, long long int nx, double *a, double *x, double *y){

    #pragma omp parallel num_threads(8)
    {
        int id = omp_get_thread_num();
        omp(na, nx, a, x, y, id);

    }
}



void my_fir_unrolling_OMP(long long int na, long long int nx, double *a, double *x, double *y, int id){
    register double *at, *yt0, *yt1, *yt2, *yt3, *xt0, *xt1, *xt2, *xt3, *xt;
    register long long int i, j;


    register long long int s = id*(nx-na)/(8);
    xt = x + s;

    for(i = 0, yt0 = y+s, yt1 = y+1+s, yt2 = y+2+s, yt3 = y+3+s; i < (nx-na)/(4*8); i++, yt0+=4, yt1+=4, yt2+=4, yt3+=4, xt+=4){
        *yt0 = 0;
        *yt1 = 0;
        *yt2 = 0;
        *yt3 = 0;
        for(j = 0, at = a, xt0 = xt, xt1 = xt+1, xt2 = xt+2, xt3 = xt+3; j < na ; j++, at++, xt0++, xt1++, xt2++, xt3++) {
            *yt0 += *xt0 * *at;
            *yt1 += *xt1 * *at;
            *yt2 += *xt2 * *at;
            *yt3 += *xt3 * *at;
        }
    }
}

void main(){

    long long int NA = 512, NX = 0x1000000;

    double *A = (double*) _aligned_malloc( NA * sizeof(double), 64 );
    double *X = (double*) _aligned_malloc( NX * sizeof(double), 64 );
    double *Y1 = (double*) _aligned_malloc( (NX - NA) * sizeof(double), 64 );
    double *Y2 = (double*) _aligned_malloc( (NX - NA) * sizeof(double), 64 );

    if( A == NULL || X == NULL || Y1 == NULL || Y2 == NULL ){
        printf("Memory Allocation Error\n\n");
        return; }

    srand( time(NULL) );
    fill( A, NA );
    fill( X, NX );

    int ref = 0;


    fflush(stdin);
        printf("\n\nDo you want to fir(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();
            firf( NA, NX, A, X, ref == 1 ? Y1 : Y2 );
            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }
    fflush(stdin);
        printf("\n\nDo you want to  my_fir_reg_point(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();
            my_fir_reg_point( NA, NX, A, X, ref == 1 ? Y1 : Y2 );
            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }
    fflush(stdin);
        printf("\n\nDo you want to  my_fir_unrolling(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();
            my_fir_unrolling( NA, NX, A, X, ref == 1 ? Y1 : Y2 );
            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }
    fflush(stdin);
        printf("\n\nDo you want to  my_fir_unrolling_vector(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();
            my_fir_unrolling_vector( NA, NX, A, X, ref == 1 ? Y1 : Y2 );
            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }
    fflush(stdin);
        printf("\n\nDo you want to  my_fir_unrolling_OMP(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();
            #pragma omp parallel num_threads(8)
            {
                int id = omp_get_thread_num();
                my_fir_unrolling_OMP( NA, NX, A, X, ref == 1 ? Y1 : Y2 , id);
            }
            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }
    fflush(stdin);
        printf("\n\nDo you want to  my_fir_unrolling_vector_OMP(y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            clock_t t = clock();

            fir( NA, NX, A, X, ref == 1 ? Y1 : Y2);

            t = clock() - t;
            printf("\n\t\t\tExecution time is %0.2f s\n\n", (float) t / CLOCKS_PER_SEC);
        }



    int n;
    if(ref == 2){
            int i;
            double *c1, *c2;
            printf("\n\nStart of Compare: ");
            for(i=0, c1=Y1, c2=Y2, n=NX - NA; i<n; i++, c1++, c2++){
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
}
