from threading import Thread, Lock, current_thread
from queue import Queue
import time
import os

print(os.cpu_count())
a = 0

def increase(lock):
    global a
    b = a
    b += 1
    a = b
    time.sleep(0.5)
    print(F"Value: {a}")


def worker(q):
    while True:
        value = q.get()
        # time.sleep(0.1)

        print(f"in {current_thread().name} got {value}")
        q.task_done()


if __name__ == "__main__":
    q = Queue()

    for i in range(10):
        t = Thread(target=worker, args=(q, ))
        t.daemon = True
        t.start()

    for j in range(20):
        q.put(j)
        # time.sleep(0.1)

    q.join()

