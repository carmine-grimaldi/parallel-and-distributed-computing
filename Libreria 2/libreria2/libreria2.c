#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/**************************Prototipi funzioni***************************/

void printVett(double *array, int dim);
double * matxvet(int m, int n, double *time, double * restrict vettore, double **matrice);

/***********************************************************************/

int main(int argc, char *argv[]) {
    int i, j, M, N; //M: Righe matrice, N: Colonne matrice
    double **matrice;
    double *risultato, *vettore;
    double time;
    
    if(argc != 3) {
    	printf("\nErrore! Occorre passare in input al PBS il numero di righe M e il numero di colonne N della matrice.\n\n");
    	exit(1);
	}
    
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    //Controlli di robustezza
    if (atoi(getenv("OMP_NUM_THREADS")) <= 0) {
        printf("\nErrore! Il numero dei threads deve essere positivo\n\n");
        exit(1);
    }
    if (M <= 0 || N <= 0) {
        printf("\nErrore! Le dimensioni dei dati di input non sono valori positivi.\n\n");
        exit(1);
    }
    
    printf("\n\nThread %d: il programma sara' eseguito con %d threads\n", omp_get_thread_num(), atoi(getenv("OMP_NUM_THREADS")));
    
    //Allocazione spazio per matrice e vettore
    matrice = (double**) malloc(M * sizeof(double*));
    for (i = 0; i < M; i++)
        matrice[i] = (double*) malloc(N * sizeof(double));
        
    vettore = (double*) malloc(N * sizeof(double));
        
    printf("\nThread %d: Matrice :\n", omp_get_thread_num());
    
    //Inizializzazione e stampa matrice
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            matrice[i][j] = (double)i + 1.;
            printf("%lf ", matrice[i][j]);
        }
        
        printf("\n");
    }
    
    printf("\n");

    //Inizializzazione vettore
    for (i = 0; i < N; i++) {
        vettore[i] = 2.;
    }
    
    printf("Thread %d: Vettore :\n", omp_get_thread_num());
    printVett(vettore, N);
    printf("\n");
    
    risultato = matxvet(M, N, &time, vettore, matrice);  
    
    //Stampa del risultato
    printf("Thread %d: Il vettore risultato e' pari a :\n", omp_get_thread_num());
    printVett(risultato, M);
    
    printf("\nIl tempo di esecuzione totale e' stato di %lf secondi.\n\n", time);
    
    /* Libero la memoria */
    
    for (i = 0; i < M; i++)
        free(matrice[i]);
        
    free(matrice);
    free(vettore);
    free(risultato);
    
    return 0;
}

/***********************************************************************/

void printVett(double *array, int dim) {
    int i;

    for (i = 0; i < dim; i++) {
        printf("\n%lf ", array[i]);
    }
    printf("\n");
}

double * matxvet(int m, int n, double *time, double * restrict vettore, double **matrice) {    
    int i, j;
    double t1, t2;
    double *risultato;
    
    //Allocazione spazio per vettore risultato
    risultato = (double*) malloc(m * sizeof(double));
    
    //Memorizzo il tempo di inizio
    t1 = omp_get_wtime();
    
    //Calcolo in parallelo del prodotto tra matrice e vettore
    #pragma omp parallel for private(i,j) shared (matrice, vettore, risultato)
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            risultato[i] += matrice[i][j] * vettore[j];
        }
    }
    
    //Memorizzo il tempo di fine
    t2 = omp_get_wtime();
    
    *time = (t2 - t1);
    
    return risultato;
}
