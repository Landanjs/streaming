# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronization primitives that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

import os
import shutil
from multiprocessing.shared_memory import SharedMemory
from time import sleep
from typing import Optional

import numpy as np
from filelock import FileLock

# Time to wait, in seconds.
TICK = 0.07

# Time out to wait before raising exception
TIMEOUT = 60


class SharedBarrier:
    """A barrier that works inter-process using a file lock and shared memory.

    We set the number of processes (and thereby initialize num_exit) on the first time this object
    is called. This is because the object is created in a per-rank process, and called by worker
    processes.

    Args:
        filelock_path (str): Path to lock file on local filesystem.
        shm_path (str): Shared memory object name in /dev/shm.
        is_local_leader (bool): Is a local leader process or not
    """

    def __init__(self, filelock_path: str, shm_path: str, is_local_leader: bool) -> None:
        self.is_local_leader = is_local_leader
        self.filelock_path = filelock_path
        self.shm_path = shm_path

        # Create three int32 fields in shared memory: num_enter, num_exit, flag.
        size = 3 * np.int32(0).nbytes

        try:
            # Creates a new shared memory block
            self._shm = SharedMemory(shm_path, True, size)
        except FileExistsError:
            sleep(TICK)
            # Attaches to an existing shared memory block
            self._shm = SharedMemory(shm_path, False, size)

        # Create filelock.
        self.dirname = os.path.dirname(filelock_path)
        os.makedirs(self.dirname, exist_ok=True)
        self.lock = FileLock(filelock_path)

        self._arr = np.ndarray(3, buffer=self._shm.buf, dtype=np.int32)
        self._arr[0] = 0
        self._arr[1] = -1
        self._arr[2] = True

    def __del__(self):
        """Destructor clears array that references shm."""
        if hasattr(self, '_shm') and self._shm is not None:
            # Close each SharedMemory instance
            self._shm.close()
            if self.is_local_leader:
                # Call unlink only once to release the shared memory
                self._shm.unlink()
            else:
                # Wait for local leader process to execute first
                sleep(1)
        if hasattr(self, 'dirname') and self.is_local_leader:
            if os.path.islink(self.dirname):
                os.unlink(self.dirname)
            shutil.rmtree(self.dirname)

    @property
    def num_enter(self) -> int:
        """Get property num_enter.

        Returns:
            int: Number of processes that have entered the barrier.
        """
        return self._arr[0]

    @num_enter.setter
    def num_enter(self, num_enter: int) -> None:
        """Set property num_enter.

        Args:
            num_enter (int): Number of processes that have entered the barrier.
        """
        self._arr[0] = num_enter

    @property
    def num_exit(self) -> int:
        """Get property num_exit.

        Returns:
            int: Number of processes that have exited the barrier.
        """
        return self._arr[1]

    @num_exit.setter
    def num_exit(self, num_exit: int) -> None:
        """Set property num_exit.

        Args:
            num_exit (int): Number of processes that have exited the barrier.
        """
        self._arr[1] = num_exit

    @property
    def flag(self) -> bool:
        """Get property flag.

        Returns:
            bool: The flag value.
        """
        return bool(self._arr[2])

    @flag.setter
    def flag(self, flag: bool) -> None:
        """Set property flag.

        Args:
            flag (bool): The flag value.
        """
        self._arr[2] = bool(flag)

    def __call__(self, num_procs: int) -> None:
        """A set number of processes enter, wait, and exit the barrier.

        Args:
            num_procs (int): How many processes are sharing this barrier.
        """
        # reinit the numpy array pointing to shared memory. Necessary when spawn is the multiprocessing method used
        self._arr = np.ndarray(3, buffer=self._shm.buf, dtype=np.int32)

        # Initialize num_exit to the number of processes.
        with self.lock:
            if self.num_exit == -1:
                self.num_exit = num_procs

        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self.num_enter:
            self.lock.release()
            while self.num_exit != num_procs:
                sleep(TICK)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == num_procs:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self.flag:
            sleep(TICK)

        # Note that we exited.
        with self.lock:
            self.num_exit += 1


def create_shared_memory(name: Optional[str] = None, size: int = 0) -> SharedMemory:
    """Create a new Shared Memory block or attach to an existing shared memory block.

    Args:
        name (str, optional): A unique shared memory block name. Defaults to None.
        size (int, optional): A size of a shared memory block. Defaults to 0.

    Returns:
        SharedMemory: An instance of shared memory block
    """
    try:
        # Creates a new shared memory block
        return SharedMemory(name, True, size)
    except FileExistsError:
        sleep(TICK)
        # Attaches to an existing shared memory block.
        return SharedMemory(name, False, size)
