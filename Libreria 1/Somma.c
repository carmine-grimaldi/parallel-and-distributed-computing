#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"


/********************************Costanti********************************/

//Quantita' oltre la quale i valori in input vengono generati casualmente
#define MAX_INPUT 20 

//Posizioni dei parametri presi in input dal main
#define ID 1
#define STRATEGIA 2
#define DIM 3
#define PRIMO_VALORE_INPUT 4

#define MAX_RAND 150
#define MIN_RAND -150


/**************************Prototipi funzioni***************************/

//Funzioni di gestione
void start(int argc, char *argv[], int idCpu, int numCpu);
int strategyChoice(int chiamante, int cpuPrint, int strat, int nCpu);
int * addendsDistribution(int idCpu, char * argv[], int * dim2, int * add);
int * sumArrayInit(int taglia, int argc, char *argv[]);
int localSum(int * addendi, int numAddendi);
int equivalentID(int idCpu, int idCpuPrint, int numCpu);
int equivalentSrcDst(int idCpu, int numCpu);
void printResults(int idCpu, int cpuPrint, int somma, double tempo);

//Strategie per sommare N numeri (interi) in parallello 
int strategy1(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t);
int strategy2(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t);
int strategy3(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t);

//Funzioni di utility
int generateRandom(int min, int max);
double logarithm(double base, double argomento);
int * calculatePowers(int base, int numPotenze);
int testPowerOfTwo(double x);

/***********************************************************************/


int main(int argc, char *argv[]) {
	int menum, numCpu;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&menum);
	MPI_Comm_size(MPI_COMM_WORLD,&numCpu);
	
	/* Controllo che il numero di elementi da sommare sia valido,
	 * in caso contrario viene segnalato l'errore e si termina il programma */
	if(atoi(argv[DIM]) < 1) {
		if(menum == atoi(argv[ID]))
		    printf("La quantita' di valori da sommare deve essere positiva!\n\n");
		MPI_Finalize();
		return 1;
	}
	else if(atoi(argv[DIM]) < numCpu) {
		if(menum == atoi(argv[ID]))
		    printf("Attenzione!\nLa quantita' di valori da sommare deve essere almeno pari al numero di CPU! (Numero di CPU: %d)\n", numCpu);
		MPI_Finalize();
		return 1;
	}
	else if(atoi(argv[ID]) < -1 || atoi(argv[ID]) > (numCpu-1)) {
		if(menum == 0)
		    printf("Attenzione!\nL'ID del processore che stampa la somma deve essere compreso tra -1 e %d (numero CPU - 1)!\n", numCpu-1);
		MPI_Finalize();
		return 1;
	}
	else {
		srand(time(NULL));
		//Se il numero di elementi e' corretto, si prosegue
		start(argc, argv, menum, numCpu);
	}
	
	MPI_Finalize();

	return 0;
}


/**************************Funzioni di Gestione**************************/

void start(int argc, char *argv[], int idCpu, int numCpu) {
	int idCpuPrint = atoi(argv[ID]);
	int idCpuMaster;
	int strategia = atoi(argv[STRATEGIA]);
	int dim, dimPersonale, risultato;
	
	int * addendi;
	int * addendiPersonali;
	
	double tempoImpiegato;
	
	/* Se in input ci viene richiesto che tutti i processori stampino
	 * la somma totale, impostiamo P0 come il processore 
	 * incaricato dello scambio dei messaggi */
	if(idCpuPrint == -1) {
		idCpuMaster = 0;
	}
	else {
		idCpuMaster = idCpuPrint;
	}
	
	if(idCpu == idCpuMaster) {
		dim = atoi(argv[DIM]);
		//Si popola l'array degli addendi
		addendi = sumArrayInit(dim, argc, argv);
	}
	
	//Comunichiamo agli altri processi (tramite una comunicazione broadcast) il numero di elementi da sommare
	MPI_Bcast(&dim, 1, MPI_INT, idCpuMaster, MPI_COMM_WORLD);
	
	//Si distribuiscono i vari elementi da sommare ad ogni processo
	addendiPersonali = addendsDistribution(idCpu, argv, &dimPersonale, addendi);
	
	int strategia_Scelta = strategyChoice(idCpu, idCpuMaster, strategia, numCpu);
	
	if(strategia_Scelta != 0) {
		//Si chiama la funzione che sommera' con una determinata politica
		if(strategia_Scelta == 1) {
			risultato = strategy1(idCpu, idCpuMaster, numCpu, addendiPersonali, dimPersonale, &tempoImpiegato);
		}
		else if(strategia_Scelta == 2) {
			risultato = strategy2(idCpu, idCpuMaster, numCpu, addendiPersonali, dimPersonale, &tempoImpiegato);
		}
		else { //strategia_Scelta == 3
			risultato = strategy3(idCpu, idCpuMaster, numCpu, addendiPersonali, dimPersonale, &tempoImpiegato);
		}
		
		//Se tutti i processi stampano il risultato, inviamo a essi il tempo totale impiegato
		if(idCpuPrint == -1) {
			MPI_Bcast(&tempoImpiegato, 1, MPI_DOUBLE, idCpuMaster, MPI_COMM_WORLD);
		}
		
		/* Se ci viene richiesto che tutti i processori stampino il risultato e la
		 * strategia richiesta è stata la prima o la seconda, facciamo recuperare
	     * il risultato completo a tutti i processori diversi da idCpuMaster (= 0) */
		if(idCpuPrint == -1 && strategia_Scelta != 3) {			
			MPI_Bcast(&risultato, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		
		printResults(idCpu, idCpuPrint, risultato, tempoImpiegato);
	}
	else {
		//La scelta effettuata non e' corretta
		MPI_Finalize();
		exit(1);
	}
}

/* Funzione che restituisce in output la strategia da applicare,
 * oppure restituisce 0 in caso di incorrettezza dell'input */
int strategyChoice(int chiamante, int cpuPrint, int strat, int nCpu) {
	if(strat == 1 || (!testPowerOfTwo(nCpu) && (strat == 2 || strat == 3))) {
		if(strat != 1 && chiamante == cpuPrint) {
			printf("Attenzione!\nCon %d processi non e' possibile applicare la strategia %d!\nSara' applicata la strategia 1.\n", nCpu, strat);
		}
		return 1;
	}
	else if(strat == 2)	{
		return 2;
	}
	else if(strat == 3)	{
		return 3;
	}
	else {
		if(chiamante == cpuPrint) {
		    printf("Attenzione!\nStrategia di somma non esistente!\n\n");
		}
		return 0;
	}
}

/* Distribuisce ad ogni processo il sottoinsieme di valori da sommare, 
 * prima di fare cio' viene allocato l'array che conterra' tali valori,
 * che viene poi restituito in output */
int * addendsDistribution(int idCpu, char * argv[], int * dim2, int * add) {
	int idCpuPrint = atoi(argv[ID]);
	int dim = atoi(argv[DIM]);
	
	int numCpu, resto, start, offset, i, tag, idSrc;
	int * v;
	
	MPI_Status status;
	
	MPI_Comm_size(MPI_COMM_WORLD, &numCpu);
	
	if(idCpuPrint == -1) {
		idCpuPrint = 0;
	}
	
	//Comunico "esternamente" quanti valori verranno inseriti nell'array
	*(dim2) = (dim / numCpu);
	resto = (dim % numCpu);
	
	/* Alcuni processi conterranno un elemento in meno nell'array,
	 * quindi aggiorniamo la loro dimensione.
	 * Cio' puo' accadere nel caso in cui il numero di elementi da sommare non
	 * e' un multiplo del numero dei processi*/
	if(equivalentID(idCpu, idCpuPrint, numCpu) < resto) {
		(*(dim2))++;
	}
	
	//Tutti i processi, tranne quello "principale", allocano il proprio array
	if(idCpu != idCpuPrint) {
		v = (int *)malloc(sizeof(int) * (*(dim2)));
	}
	
	if(idCpu == idCpuPrint)	{
		/* Il processo principale leggera' i valori da sommare 
		 * dall'array principale, poiche' e' una sua copia personale */
		v = add;
		
		offset = *(dim2);
		start = 0;
		
		/* Il processo principale invia i valori da sommare ad ogni processo (escluso se stesso)
		 * ad intervalli che iniziano in start e terminano in offset - 1 */
		for(i = 1; i < numCpu; i++) {
			/* Per calcolare l'id del destinatario della somma parziale si passa idCpu + i, poiche' alla prima
			 * iterazione si inoltra la somma parziale a idCpuPrint + 1, alla seconda iterazione a 
			 * idCpuPrint + 2 e cosi via */
			idSrc = equivalentSrcDst(i+idCpu, numCpu);
			start += offset;
			tag = idSrc + 100;
			
			if(i == resto) {
				offset--;
			}
			
			MPI_Send(&add[start], offset, MPI_INT, idSrc, tag, MPI_COMM_WORLD);
		}
	}
	else {
		/* Tutti i processi, tranne quello principale, ricevono i valori 
		 * da sommare e li memorizzano in un array personale */
		tag = 100 + idCpu;
		MPI_Recv(v, (*(dim2)), MPI_INT, idCpuPrint, tag, MPI_COMM_WORLD, &status);
	}
	
	return v;
}

/* Questa funzione alloca un array per poi riempirlo.
 * Se si devono sommare piu' di 20 valori si riempie con valori casuali (oppure tutti 1),
 * altrimenti viene riempito con valori presi in input dal main */
int * sumArrayInit(int taglia, int argc, char * argv[]) {
	int * v = (int *)malloc(sizeof(int) * taglia);
	int i;
	
	if(taglia <= MAX_INPUT)	{
		printf("\nValori da sommare: \n");
		for(i = PRIMO_VALORE_INPUT; i < argc; i++) {
			v[(i - PRIMO_VALORE_INPUT)] = atoi(argv[i]);
			printf("%d\t", v[(i - PRIMO_VALORE_INPUT)]);
		}
	}
	else {
		//printf("\nValori da sommare: \n");
		for(i = 0; i < taglia; i++)	{
			//v[i] = generateRandom(MIN_RAND, MAX_RAND);
			v[i] = 1;
			//printf("%d\t", v[i]);
		}
	}
	//printf("\n\n");

	return v;
}

//Funzione che effettua la somma di N valori interi
int localSum(int * addendi, int numAddendi) {
	int sommaParziale = 0, i;
	
	for(i = 0; i < numAddendi; i++)	{
		sommaParziale += addendi[i];
	}
	
	return sommaParziale;
}

/* Presi in input il numero del processo, il numero del processo che deve 
 * "stampare" e la quantita' di processi in esecuzione, rende idCpuPrint 
 * equivalente all'id che avrebbe tale processo se il processo principale fosse zero. */
int equivalentID(int idCpu, int idCpuPrint, int numCpu) {
	return ((idCpu - idCpuPrint + numCpu) % numCpu);
}

/* Preso in input l'id di un processo che dovrebbe inviare o ricevere un messaggio,
 * produce in output l'id del processo che effettivamente dovra' effettuare tale operazione */
int equivalentSrcDst(int idCpu, int numCpu) {
	return ((numCpu + idCpu) % numCpu);
}

void printResults(int idCpu, int cpuPrint, int somma, double tempo) {
	if(idCpu == cpuPrint || cpuPrint <= -1) {
		if(cpuPrint <= -1 && idCpu != 0) {
			printf("\n\n\tIl processo P%d ha calcolato la somma %d\n", idCpu, somma);
		}
		else {
		    printf("\n\n\tIl processo P%d ha calcolato la somma %d in %f secondi\n", idCpu, somma, tempo);
		}
	}
}


/**********************Funzioni di somma parallela***********************/

int strategy1(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t) {
	int sommaParziale;
	int sommaTotale, tag, idSrc; 
	
	double t0, t1, time;
	
	MPI_Status status;
	
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	
	sommaParziale = localSum(addendi, numAddendi);
	
	/* Nel primo ramo dell'if ci entreranno tutti i processi tranne quello principale,
	 * essi comunicheranno la loro somma parziale al processo principale, che la sommera' alla propria */
	if(chiamante != cpuPrint) {
		tag = 200 + chiamante;
		MPI_Send(&sommaParziale, 1, MPI_INT, cpuPrint, tag, MPI_COMM_WORLD);
	}
	else {
		int i;
		sommaTotale = sommaParziale;
		for(i = 1; i < numCpu; i++)	{
			idSrc = equivalentSrcDst(i + chiamante, numCpu);
			tag = 200 + idSrc;
			MPI_Recv(&sommaParziale, 1, MPI_INT, idSrc, tag, MPI_COMM_WORLD, &status);
			sommaTotale += sommaParziale;
		}
	}
	
	t1=MPI_Wtime();
	time = t1 - t0;
	MPI_Reduce(&time, t, 1, MPI_DOUBLE, MPI_MAX, cpuPrint, MPI_COMM_WORLD);
	
	if(chiamante == cpuPrint) {
	    return sommaTotale;
	}
	else {
		return sommaParziale;
	}
}

int strategy2(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t) {
	int sommaParziale;
	int sommaTmp = 0, tag, i, shiftId;
	double log2nCpu;
	
	int * potenzeDi2 = NULL;
	
	double t0, t1, time;
	
	MPI_Status status;
	
	/* Un solo processo calcola i dati che verrano utilizzati per la somma parallela,
	 * questi dati vengono calcolati in anticipo per non aggiungere overhead alla somma. */ 
	if(chiamante == cpuPrint) {
		log2nCpu = logarithm(2, numCpu);
		potenzeDi2 = calculatePowers(2, (log2nCpu + 1));
	}
	
	//Al termine il processore che li ha calcolati li spedisce agli altri processori
	MPI_Bcast(&log2nCpu, 1, MPI_INT, cpuPrint, MPI_COMM_WORLD);
	
	//Prima di ricevere le potenze di due dal processo principale, i processi devono allocare il proprio array
	if(chiamante != cpuPrint) {
		potenzeDi2 = (int *) malloc(sizeof(int) * (log2nCpu + 1));
	}
	MPI_Bcast(potenzeDi2, (log2nCpu + 1), MPI_INT, cpuPrint, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	
	sommaParziale = localSum(addendi, numAddendi);
	
	/* I processi a coppie calcolano una somma parziale, ad ogni iterazione il numero di coppie si dimezza, 
	 * costruendo un "albero capovolto" dove le foglie sono i valori in input e la somma e' la radice dell'albero,
	 * mentre i nodi interni sono le somme parziali calcolate per arrivare alla somma finale */
	for(i = 0; i < log2nCpu; i++) {
		shiftId = equivalentID(chiamante, cpuPrint, numCpu);
		if(shiftId % potenzeDi2[i] == 0) {
			if(shiftId % potenzeDi2[i + 1] != 0) {
				tag = 300 + chiamante;
				MPI_Send(&sommaParziale, 1, MPI_INT, equivalentSrcDst((chiamante - potenzeDi2[i]), numCpu), tag, MPI_COMM_WORLD);
			}
			else {
				sommaTmp = sommaParziale;
				tag = 300 + equivalentSrcDst((chiamante + potenzeDi2[i]), numCpu);
				MPI_Recv(&sommaParziale , 1, MPI_INT, equivalentSrcDst(chiamante + potenzeDi2[i], numCpu), tag, MPI_COMM_WORLD, &status);
				sommaParziale += sommaTmp;
			}
		}
	}
	
	t1=MPI_Wtime();
	time = t1 - t0;
	MPI_Reduce(&time, t, 1, MPI_DOUBLE, MPI_MAX, cpuPrint, MPI_COMM_WORLD);
	
	if(potenzeDi2 != NULL) {
		free(potenzeDi2);
	}
	
	return sommaParziale;
}

int strategy3(int chiamante, int cpuPrint, int numCpu, int * addendi, int numAddendi, double * t) {
	int sommaParziale;
	int sommaTmp = 0, tag, i;
	double log2nCpu;
	
	double t0, t1, time;
	
	int * potenzeDi2 = NULL;
	
	MPI_Status status;
	
	/* Un solo processo calcola i dati che verrano utilizzati per la somma parallela,
	 * questi dati vengono calcolati in anticipo per non aggiungere overhead alla somma. */ 
	if(chiamante == cpuPrint) {
		log2nCpu = logarithm(2, numCpu);
		potenzeDi2 = calculatePowers(2, (log2nCpu + 1));
	}
	
	//Al termine, il processore che ha calcolato i dati, li spedisce agli altri processori
	MPI_Bcast(&log2nCpu, 1, MPI_INT, cpuPrint, MPI_COMM_WORLD);
	
	if(chiamante != cpuPrint) {
		potenzeDi2 = (int *) malloc(sizeof(int) * (log2nCpu + 1));
	}
	MPI_Bcast(potenzeDi2, (log2nCpu + 1), MPI_INT, cpuPrint, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	
	sommaParziale = localSum(addendi, numAddendi);
	
	/* Come per la strategia 2 ma in questo caso c'e' un doppio scambio di messaggi in ingresso e uscita,
	 * al termine tutti i processi avranno una propria copia della somma, al contrario della strategia 2 */
	for(i = 0; i < log2nCpu; i++) {
		if((equivalentID(chiamante, cpuPrint, numCpu) % potenzeDi2[i + 1]) < potenzeDi2[i]) {
			tag = ((i*1000) + (400 + chiamante));
			MPI_Send(&sommaParziale, 1, MPI_INT, equivalentSrcDst((chiamante + potenzeDi2[i]), numCpu), tag, MPI_COMM_WORLD);
			
			tag = ((i*1000) + (500 + equivalentSrcDst((chiamante + potenzeDi2[i]), numCpu)));
			MPI_Recv(&sommaTmp , 1, MPI_INT, equivalentSrcDst((chiamante + potenzeDi2[i]), numCpu), tag, MPI_COMM_WORLD, &status);
			
			sommaParziale += sommaTmp;
		}
		else {	
			tag = ((i*1000) + (400 + equivalentSrcDst((chiamante - potenzeDi2[i]), numCpu)));
			MPI_Recv(&sommaTmp , 1, MPI_INT, equivalentSrcDst((chiamante - potenzeDi2[i]), numCpu), tag, MPI_COMM_WORLD, &status);
			
			tag = ((i*1000 +(500 + chiamante)));
			MPI_Send(&sommaParziale, 1, MPI_INT, equivalentSrcDst((chiamante - potenzeDi2[i]), numCpu), tag, MPI_COMM_WORLD);
			
			sommaParziale += sommaTmp;
		}	
	}
	
	t1=MPI_Wtime();
	time = t1 - t0;
	MPI_Reduce(&time, t, 1, MPI_DOUBLE, MPI_MAX, cpuPrint, MPI_COMM_WORLD);
	
	if(potenzeDi2 != NULL)	{
		free(potenzeDi2);
	}
	
	return sommaParziale;
}


/***************************Funzioni di utility **************************/

//Funzione che alloca un array e lo riempie con potenze contigue partendo da base^0
int * calculatePowers(int base, int numPotenze) {
	int * potenze = (int *)malloc(sizeof(int) * numPotenze);
	int i;
	
	potenze[0] = 1;
	for(i = 1; i < numPotenze; i++)	{
		potenze[i] = potenze[i - 1] * base;
	}
	
	return potenze;
}

//Funzione che calcola il logarithm in una qualsiasi base positiva
double logarithm(double base, double argomento) {
	return (log10(argomento)/log10(base));
}

//Funzione che controlla se un valore e' una potenza di due
int testPowerOfTwo(double x) {
	return ((logarithm(2, x) - (floor(logarithm(2, x)))) == 0);
}

//Generazione di valori casuali compresi nell'intervallo [min, max]
int generateRandom(int min, int max) {
	return ((rand() % (max - min + 1)) + min);
}

/************************************************************************/
