
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

#define SIZE 100
int array[SIZE];

int main(int argc, char *argv[])
{
     int rank, nprocs;
     MPI_Win win;
     
     MPI_Init(&argc, &argv);
     
     MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
     
     MPI_Win_create(array, SIZE*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
     
     MPI_Win_lock(MPI_LOCK_EXCLUSIVE, MPI_PROC_NULL, 0, win);
     MPI_Win_unlock(MPI_PROC_NULL, win);
     
     MPI_Win_lock(MPI_LOCK_SHARED, MPI_PROC_NULL, 0, win);
     MPI_Win_unlock(MPI_PROC_NULL, win);
        
     MPI_Win_free(&win);
     
     MPI_Finalize();

     if (rank == 0) printf(" No errors\n");

     return 0;
}
