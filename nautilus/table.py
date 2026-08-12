import numpy as np


class Table:
    """A lightweight table."""

    def __init__(self, **kwargs):
        """Create a table.

        Parameters
        ----------
        **kwargs
            Arrays used to initialize the table. Must all have the same length.

        Raises
        ------
        ValueError
            If arrays do not have the same length or no data is passed.

        """
        if len(kwargs) == 0:
            msg = "`Table` object needs data."
            raise ValueError(msg)

        for i, arr in enumerate(kwargs.values()):
            if i == 0:
                n = len(arr)
            elif n != len(arr):
                msg = "All arrays must have the same length."
                raise ValueError(msg)

        self.n = n
        self.n_max = n
        self.data = kwargs

    def __getitem__(self, key):
        """Get a sample by column or row(s).

        Parameters
        ----------
        key : str, slice, or numpy.ndarray
            Key, slice, or array.

        Returns
        -------
        result : numpy.ndarray or desilike.statistics.Samples
            If ``key`` is a ``str``, the value, i.e., column, for that key.
            If ``key`` is a slice or array, a new ``Table`` object
            corresponding to those rows.

        Raises
        ------
        TypeError
            If ``key`` is not a string, slice, or filter array.

        """
        if isinstance(key, str):
            return self.data[key][:self.n]
        elif isinstance(key, (slice, np.ndarray)):
            rows = key
            return self.__class__(**{
                key: self.data[key][:self.n][rows] for key in self.data})
        else:
            msg = "Data can only be accessed via strings, slices, or arrays."
            raise TypeError(msg)

    def __len__(self):
        """Return the number of rows."""
        return self.n

    def _extend(self, k):
        for key, arr in self.data.items():
            pad_width = [(0, 0)] * arr.ndim
            pad_width[0] = (0, k)
            self.data[key] = np.pad(arr, pad_width, mode="empty")
        self.n_max += k

    def append(self, table):
        """Append a table, i.e., add additional rows at the bottom.

        Parameters
        ----------
        table : nautilus.table.Table
            Table to add. Must have the same keys as the current table.

        Raises
        ------
        ValueError
            If keys do not match.

        """
        if set(self.data.keys()) != set(table.data.keys()):
            msg = "Keys do not match."
            raise ValueError(msg)

        if self.n_max < len(self) + len(table):
            self._extend(max(10000, len(self) + len(table) - self.n_max))

        for key in self.data:
            self.data[key][self.n:self.n + len(table)] = table[key]

        self.n += len(table)
