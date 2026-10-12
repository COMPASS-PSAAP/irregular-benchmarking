/* PMPI wrapper (LD_PRELOAD), diagnostic only. Reports at MPI_Finalize, as max per
 * rank and total:
 *   self:  MPI_Sendrecv on MPI_COMM_SELF (locality_aware's per-element copy loop
 *          after ce69e90)
 *   p2p:   Isend/Irecv/Send_init/Recv_init on a comm other than MPI_COMM_WORLD
 *          (MPI Advance's own communicators; plain MPI code in ABCG uses WORLD) */
#include <mpi.h>
#include <stdio.h>

static long n_self = 0, n_p2p = 0;

static void count_p2p(MPI_Comm comm)
{
    if (comm != MPI_COMM_WORLD) n_p2p++;
}

int MPI_Sendrecv(const void* sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                 void* rbuf, int rcount, MPI_Datatype rtype, int source, int rtag,
                 MPI_Comm comm, MPI_Status* status)
{
    if (comm == MPI_COMM_SELF) n_self++;
    return PMPI_Sendrecv(sbuf, scount, stype, dest, stag, rbuf, rcount, rtype, source, rtag,
                         comm, status);
}

int MPI_Isend(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm,
              MPI_Request* req)
{
    count_p2p(comm);
    return PMPI_Isend(buf, count, type, dest, tag, comm, req);
}

int MPI_Irecv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm,
              MPI_Request* req)
{
    count_p2p(comm);
    return PMPI_Irecv(buf, count, type, source, tag, comm, req);
}

int MPI_Send_init(const void* buf, int count, MPI_Datatype type, int dest, int tag,
                  MPI_Comm comm, MPI_Request* req)
{
    count_p2p(comm);
    return PMPI_Send_init(buf, count, type, dest, tag, comm, req);
}

int MPI_Recv_init(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm,
                  MPI_Request* req)
{
    count_p2p(comm);
    return PMPI_Recv_init(buf, count, type, source, tag, comm, req);
}

int MPI_Finalize(void)
{
    int rank;
    long v[2] = {n_self, n_p2p}, mx[2], sum[2];
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Reduce(v, mx, 2, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    PMPI_Reduce(v, sum, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("COUNTS self_sendrecv: max/rank %ld total %ld | non-WORLD p2p: max/rank %ld total %ld\n",
               mx[0], sum[0], mx[1], sum[1]);
    return PMPI_Finalize();
}
