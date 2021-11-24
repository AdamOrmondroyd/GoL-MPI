from mpi4py import MPI
import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt
import sys

# Function that advances the game one step
def step(cells):
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    count = scs.convolve2d(cells, mask, mode="same", boundary="wrap")
    newcells = np.where(
        cells == 1,
        np.where((count > 1), np.where(count < 4, 1, 0), 0),
        np.where(count == 3, 1, 0),
    )
    return newcells


# Set up communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

# 12 is so there is space for the cells from other ranks after communication
cells = np.zeros((10, 12))
if rank == 0:
    cells[3, 6:8] = 1
    cells[4, 4:6] = 1
    cells[4, 7:9] = 1
    cells[5, 4:8] = 1
    cells[6, 5:7] = 1

for i in range(60):
    cells = step(cells)  # step before
    #################################################
    #    Communication of boundaries goes here      #
    #################################################
    left_to_send = cells[:, 1].copy()
    right_to_send = cells[:, -2].copy()
    if rank == 0:
        print(right_to_send)
    new_left = np.empty(10, dtype="int")
    new_right = np.empty(10, dtype="int")
    # send left, receive right
    comm.Sendrecv(
        left_to_send,
        dest=(rank - 1) % size,
        recvbuf=new_right,
        source=(rank + 1) % size,
    )
    comm.Sendrecv(
        right_to_send,
        dest=(rank + 1) % size,
        recvbuf=new_left,
        source=(rank - 1) % size,
    )
    if rank == 1:
        print(new_left)
    cells[:, 0] = new_left
    cells[:, -1] = new_right

    # cells = step(cells) # step after

# Plot results, duck should be in 4th rank
fig = plt.figure()
ax = plt.axes()
# ax.matshow(cells[:, 1:11])
ax.matshow(cells)
filename = "PlotRank" + str(rank) + ".png"
plt.savefig(filename)
print(f"[{rank}]")
print(cells)
