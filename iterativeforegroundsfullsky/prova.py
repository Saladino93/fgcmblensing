from plancklens.helpers import mpi
import numpy as np

print(mpi.rank)
d = "/home/users/d/darwish/scratch/"

if __name__ == '__main__':
    if mpi.rank == 0:
        print("Rank 0!!")
        np.savetxt(d+"a.txt", np.arange(10))
    mpi.barrier()

    np.loadtxt(d+"a.txt")
