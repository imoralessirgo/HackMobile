from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = [1,2,3,4,5]
    comm.send(data, dest = 1)
    
elif rank == 1:
    data = comm.recv(source = 0)
    data = data[::-1]
    comm.send(data,dest = 0)

if rank == 0:
    data = comm.recv(source = 1)


