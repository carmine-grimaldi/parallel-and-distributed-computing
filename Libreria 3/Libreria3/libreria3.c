#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/*******************************Strutture*******************************/

typedef struct {
    MPI_Comm grid_comm; /* communicator di griglia globale  */
    MPI_Comm row_comm;  /* communicator di righe della griglia */
    MPI_Comm col_comm;  /* communicator di colonne della griglia */
    int n_proc;         /* numero di processori */
    int grid_dim;       /* dimensione della griglia, = sqrt(n_proc) */
    int my_row;         /* posizione di riga di un processo in una griglia */
    int my_col;         /* posizione di colonna di un processo in una griglia */
    int my_rank;        /* il rank nella griglia */
} GridInfo;

/**************************Prototipi funzioni***************************/

/* Inizializzazione per la griglia dei processi */
void grid_init(GridInfo *grid);

/* Funzioni per operazioni su matrici */
void matrix_creation(double **pA, double **pB, double **pC, int size);
void matrix_init(double *A, double *B, int size);
void matrix_dot(double *A, double *B, double *C, int n);
void matrix_print(double *A, int n);
void matrix_free(double **pA, double **pB, double **pC);

/* Algoritmo BMR: prodotto scalare parallelo tra matrici */
void BMRAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid);

/***********************************************************************/


int main(int argc, char *argv[]) {
    int i, j;
    double *pA, *pB, *pC;
    double *local_pA, *local_pB, *local_pC;
    int matrix_size;
    
    MPI_Init(&argc, &argv);
    
    GridInfo grid;
    grid_init(&grid);
    
    if(argc != 2) {
        if(grid.my_rank == 0) {
            printf("Errore: occorre passare in input il numero \"M\" di righe e di colonne della matrice.\n");
            printf("Sintassi di invocazione PBS: qsub nome_file.pbs -v M=[matrix_size]\n");
        }
        
        MPI_Finalize();
        exit(1);
    }
    
    matrix_size = atoi(argv[1]);
    
    /* Controlli di errore */
    if(matrix_size < 1) {
		if(grid.my_rank == 0)
            printf("Errore: matrix_size (=%d) deve essere un numero positivo!\n\n", matrix_size, grid.n_proc);
        MPI_Finalize();
        exit(1);
    }
    if (matrix_size % grid.grid_dim != 0) {
		if(grid.my_rank == 0)
            printf("Errore: la dimensione della matrice (=%d) non e' divisibile\n        "
		           "per la dimensione della griglia dei processi (=%d)!\n\n", matrix_size, grid.n_proc);
        MPI_Finalize();
        exit(1);
    }
    
    if (grid.my_rank == 0) {
        matrix_creation(&pA, &pB, &pC, matrix_size);
        matrix_init(pA, pB, matrix_size);
        
        if(matrix_size <= 10) {
            printf("Matrice A:\n"); 
            matrix_print(pA, matrix_size);
            printf("Matrice B:\n"); 
            matrix_print(pB, matrix_size);
        }
    }
    
    int local_matrix_size = matrix_size / grid.grid_dim;
    matrix_creation(&local_pA, &local_pB, &local_pC, local_matrix_size);
    
    MPI_Datatype blocktype, type;
    int array_size[2] = {matrix_size, matrix_size};
    int subarray_sizes[2] = {local_matrix_size, local_matrix_size};
    int array_start[2] = {0, 0};
    
    MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start,
                             MPI_ORDER_C, MPI_DOUBLE, &blocktype);
    MPI_Type_create_resized(blocktype, 0, local_matrix_size * sizeof(double), &type);
    MPI_Type_commit(&type);
    
    int displs[grid.n_proc];
    int sendcounts[grid.n_proc];
    
    if (grid.my_rank == 0) {
        for (i = 0; i < grid.n_proc; ++i) {
            sendcounts[i] = 1;
        }
        int disp = 0;
        for (i = 0; i < grid.grid_dim; ++i) {
              for (j = 0; j < grid.grid_dim; ++j) {
                    displs[i * grid.grid_dim + j] = disp;
                    disp += 1;
            }
            disp += (local_matrix_size - 1) * grid.grid_dim;
        }
    }
    
    MPI_Scatterv(pA, sendcounts, displs, type, local_pA,
                 local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pB, sendcounts, displs, type, local_pB,
                 local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time, end_time;
    MPI_Barrier(grid.grid_comm);
	
    if (grid.my_rank == 0) {
        start_time = MPI_Wtime();
    }
    
    BMRAlgorithm(local_pA, local_pB, local_pC, local_matrix_size, &grid);
    
    /* Raccolgo le sottomatrici da tutti i processi */
    MPI_Gatherv(local_pC, local_matrix_size*local_matrix_size, MPI_DOUBLE, pC, sendcounts, displs, type, 0, MPI_COMM_WORLD);
	    
    if (grid.my_rank == 0) {
        end_time = MPI_Wtime() - start_time;
		
        if(matrix_size <= 10) {
            printf("Matrice C (risultato prodotto scalare tra A e B):\n");
            matrix_print(pC, matrix_size);
        }    
        
        printf("---~---~---~---~---~---~---~---~---~---\n");
        printf("Tempo algoritmo BMR: %.6lf secondi\nNumero di processi: %d\nDimensione matrice: %d\n", end_time, grid.n_proc, matrix_size);
        printf("---~---~---~---~---~---~---~---~---~---\n\n");

        matrix_free(&pA, &pB, &pC);
    }
    
    matrix_free(&local_pA, &local_pB, &local_pC);
    MPI_Finalize();
    return 0;
}


/***********************************************************************/

/* Inizializza la griglia dei processi */
void grid_init(GridInfo *grid) {
    int old_rank;
    int dimensions[2];
    int periods[2];
    int coordinates[2];
    int free_coords[2];

    /* Raccolgo le informazioni generali prima di procedere */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->n_proc));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    grid->grid_dim = (int)sqrt(grid->n_proc);
    
    /* Controllo di robustezza */
    if (grid->grid_dim * grid->grid_dim != grid->n_proc) {
        if (old_rank == 0)
            printf("Errore: \'-np %d\' non e' un quadrato perfetto!\n", grid->n_proc);
        
        MPI_Finalize();
        exit(1);
    }
    
    /* Inizializzo le dimensioni */
    dimensions[0] = dimensions[1] = grid->grid_dim;
    periods[0] = periods[1] = 1; /* Caso griglia periodica */

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &(grid->grid_comm));
    
    /* Siccome abbiamo impostato il parametro reorder a true, cio' potrebbe aver cambiato i ranks */
    MPI_Comm_rank(grid->grid_comm, &(grid->my_rank));
    
    /* Raccolgo le coordinate cartesiane per il processo corrente */
    MPI_Cart_coords(grid->grid_comm, grid->my_rank, 2, coordinates);
    
    /* Memorizzo i valori delle coordinate per il processo corrente */
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* Creo communicator di righe */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

    /* Creo communicator di colonne */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));
}


/* Alloca uno spazio di memoria quadratico rispetto a "size", per ogni matrice data in input */
void matrix_creation(double **pA, double **pB, double **pC, int size) {
    *pA = (double *)malloc(size * size * sizeof(double));
    *pB = (double *)malloc(size * size * sizeof(double));
    *pC = (double *)calloc(size * size, sizeof(double));
}


/* Inizializza la matrice con valori predefiniti */
void matrix_init(double *A, double *B, int size) {
    int i;
    for (i = 0; i < size * size; ++i) {
        *(A + i) = (double)i + 1.;
        *(B + i) = (double)i + 2.;
    }
}


/* Effettua il prodotto scalare tra le matrici A e B, e lo memorizza in C */
void matrix_dot(double *A, double *B, double *C, int size) {
    int i, j, k;
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            for (k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}


/* Stampa la matrice di valori reali data in input */
void matrix_print(double *A, int size) {
    int i;
    
    if(size > 5)
        printf("---~---~---~---~---~---~---~---~---~---~---~---~---~---\n");
    else
        printf("---~---~---~---~---~---~---~---\n");
        
    
    for (i = 0; i < size * size; ++i) {
        printf("%.2lf ", *(A + i));
        if ((i + 1) % size == 0){
            printf("\n");
        }
    }
    
    if(size > 5)
        printf("---~---~---~---~---~---~---~---~---~---~---~---~---~---\n\n");
    else
        printf("---~---~---~---~---~---~---~---\n\n");
}


/* Libera la memoria per tutte le matrici date in input */
void matrix_free(double **pA, double **pB, double **pC) {
    free(*pA);
    free(*pB);
    free(*pC);
}


/* ------------------------------ Algoritmo BMR ------------------------------ */
void BMRAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid) {
    MPI_Status status;
    int root;
	
	/* Alloco lo spazio di memoria per il blocco di A da trasmettere (broadcast) */
    double *buff_A = (double*)calloc(size * size, sizeof(double));
    
    // Calcolo gli indirizzi per lo spostamento circolare di B
    int src = (grid->my_row + 1) % grid->grid_dim;
    int dst = (grid->my_row - 1 + grid->grid_dim) % grid->grid_dim;

    /**
     * Ad ogni iterazione:
     *   1. trova i blocchi che si trovano sulla diagonale della griglia dei processi
     *   2. condividi quel blocco con la riga della griglia a cui appartiene quel processo
     *   3. moltiplica l'A aggiornato (o buff_A) con B e salvalo in C
     *   4. sposta i blocchi di B verso la riga precedente della stessa colonna
     */
    int stage;
    for (stage = 0; stage < grid->grid_dim; ++stage) {
        root = (grid->my_row + stage) % grid->grid_dim;
        if (root == grid->my_col) {
            MPI_Bcast(A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_dot(A, B, C, size);
        } else {
            MPI_Bcast(buff_A, size * size, MPI_DOUBLE, root, grid->row_comm);
            matrix_dot(buff_A, B, C, size);
        }
        MPI_Sendrecv_replace(B, size * size, MPI_DOUBLE, dst, 0, src, 0, grid->col_comm, &status);
    }
}

/* -------------------------------------------------------------------------- */
