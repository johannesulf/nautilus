"""Module implementing helper functions for working with pools."""


def initialize_worker(likelihood):
    """
    Initialize a worker for likelihood evaluations.

    Parameters
    ----------
    likelihood : function
        Likelihood function that each worker will evaluate.
    """
    global LIKELIHOOD
    LIKELIHOOD = likelihood


def likelihood_worker(*args):
    """
    Have the worker evaluate the likelihood.

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


def pool_size(pool):
    """
    Determine the size of a pool, i.e., how many workers it has.

    Parameters
    ----------
    pool : object
        The pool object.

    Returns
    -------
    size : int
        The size of the pool.

    Raises
    ------
    ValueError
        If the pool size cannot be determined.

    """
    for attr in ['_processes', '_max_workers', 'size']:
        if hasattr(pool, attr):
            return getattr(pool, attr)
    raise ValueError('Cannot determine size of pool.')