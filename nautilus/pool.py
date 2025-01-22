"""Module implementing helper functions for working with pools."""

from multiprocessing import Pool


def initialize_worker(likelihood):
    """Initialize a worker for likelihood evaluations.

    Parameters
    ----------
    likelihood : function
        Likelihood function that each worker will evaluate.

    """
    global LIKELIHOOD
    LIKELIHOOD = likelihood


def likelihood_worker(*args):
    """Have the worker evaluate the likelihood.

    Parameters
    ----------
    *args : tuple
        Arguments to be passed to the likelihood function.

    Returns
    -------
    object
        Return value of the likelihood function.

    """
    return LIKELIHOOD(*args)


class NautilusPool:
    """Wrapper for avoiding implementation-specific details elsewhere.

    Attributes
    ----------
    pool : object or None
        Pool used for parallelization.

    """

    def __init__(self, pool, likelihood=None):
        """Initialize a pool.

        Parameters
        ----------
        pool : object
            Pool used for parallelization. If a number, initialize a pool
            from the `multiprocessing` library with the specified number of
            workers.
        likelihood : function, optional
            Likelihood function to cache. Defaul is None.

        """
        if isinstance(pool, int):
            self.pool = Pool(pool, initializer=initialize_worker,
                             initargs=(likelihood, ))
        else:
            self.pool = pool

    def map(self, func, iterable):
        """Loop a function over an iterable like the built-in map function.

        Parameters
        ----------
        func : function
            Function to use.
        iterable : object
            Iterable object such as a list or numpy array.

        Returns
        -------
        result : list
            Result of applying the function to all values in the iterable.

        """
        if 'distributed.client.Client' in str(type(self.pool)):
            return list(self.pool.gather(self.pool.map(func, iterable)))
        else:
            return list(self.pool.map(func, iterable))

    @property
    def size(self):
        """Determine the number of workers in the pool.

        Returns
        -------
        size : int
            Size of the pool.

        Raises
        ------
        ValueError
            If the pool size cannot be determined.


        """
        if 'distributed.client.Client' in str(type(self.pool)):
            return len(self.pool.nthreads())
        for attr in ['_processes', '_max_workers', 'size', 'nt']:
            if hasattr(self.pool, attr):
                return getattr(self.pool, attr)
        raise ValueError('Cannot determine size of pool.')
