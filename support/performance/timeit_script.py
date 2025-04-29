import sys
from timeit import timeit

from multiprocessing import Pool
from subprocess import run, PIPE


def run_command(command):
    process = run("python " + command, shell=True, stdout=PIPE, stderr=PIPE)
    # now wait for the command to finish
    if process.returncode != 0:
        raise RuntimeError(
            f"Command '{command}' failed with error: {process.stderr.decode()}"
        )


def timeit_command(command, number=1):
    """
    Time the execution of a shell command using timeit.

    Parameters
    ----------
    command : str
        The shell command to execute.
    number : int, optional
        The number of times to execute the command. Default is 1.

    Returns
    -------
    float
        The average time taken to execute the command.
    """
    return timeit(lambda: run_command(command), number=number) / number


def timeit_multiprocess_command(command, processes=4, number=1):
    """
    Time the execution of a shell command using timeit with multiprocessing.

    Parameters
    ----------
    command : str
        The shell command to execute.
    processes : int, optional
        The number of processes to use. Default is 4.
    number : int, optional
        The number of times to execute the command. Default is 1.

    Returns
    -------
    float
        The average time taken to execute the command.
    """
    with Pool(processes) as pool:
        return (
            timeit(lambda: pool.map(run_command, [command] * number), number=1) / number
        )


if __name__ == "__main__":
    command = sys.argv[1]
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    print(f"Timing command: {command}")
    print(f"Number of executions: {number}")
    print(f"Number of processes: {processes}")

    if processes > 1:
        time = timeit_multiprocess_command(command, processes, number)

    else:
        time = timeit_command(command, number)

    print(f"Average time taken: {time:.4f} seconds")
