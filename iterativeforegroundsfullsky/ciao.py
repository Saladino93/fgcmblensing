from plancklens.helpers import mpi

jobs = [1, 2, 3, 4, 5, 6]

for idx in jobs[mpi.rank::mpi.size]:
    print(mpi.rank, idx)
