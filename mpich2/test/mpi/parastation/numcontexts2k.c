
#include <mpi.h>

#define NUM_COMMS (2 * 1024 - 2)

/* 2k is the default for the max number of contexts.
   (see mpich2/src/include/mpiimpl.h)
   Therefore, it's unlikely that this test fails.
*/

int main(int argc, char* argv[])
{
	int i;
	int world_rank, rank[NUM_COMMS];
	int world_size, size[NUM_COMMS];
	MPI_Comm comm_array[NUM_COMMS];

	MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	for(i=0; i<NUM_COMMS; i++) {
		
		int color = (world_rank + 1) % world_size;

		MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm_array[i]);

		MPI_Comm_rank(comm_array[i], &rank[i]);
		MPI_Comm_size(comm_array[i], &size[i]);
	}

	for(i=0; i<NUM_COMMS; i++) {
		MPI_Comm_free(&comm_array[i]);
	}

	printf(" No errors\n");

	MPI_Finalize();
}
