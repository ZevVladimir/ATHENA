#from dask_mpi import initialize
import os
import dask.array as da
from dask.distributed import Client
import time
from contextlib import contextmanager
from distributed.scheduler import logger
import socket
from dask_cuda import LocalCUDACluster

@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

def example_function():
    print(f"start example")
    x = da.random.random((100, 100, 10), chunks=(10, 10, 5))
    print(0)
    y = da.random.random((100, 100, 10), chunks=(10, 10, 5))
    print(1)
    z = (da.arcsin(x) + da.arccos(y)).compute()
    print(z)

    

if __name__ == "__main__":
    #initialize(worker_class="dask_cuda.CUDAWorker", local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/", interface="ib0")
    #initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/", interface="ib0", nthreads=int(os.environ['SLURM_CPUS_PER_TASK']))
    #client = Client()
    cluster = LocalCUDACluster()
    client = Client(cluster)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node
    
    print()
    logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    print()

    with timed("test"):
        example_function()
