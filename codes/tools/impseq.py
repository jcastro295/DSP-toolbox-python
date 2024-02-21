"""

impseq.py
---------

"""

def impseq(n0, n1, n2):
    """
    Generates x(n) = delta(n - n0); n1 <= n, n0 <= n2
    """

    if n0 < n1 or n0 > n2 or n1 > n2:
        raise ValueError("Arguments must satisfy n1 <= n0 <= n2")

    n = list(range(n1, n2 + 1))
    x = [(1 if n_i == n0 else 0) for n_i in n]
    
    return x, n
