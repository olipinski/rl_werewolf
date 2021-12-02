import csv
import math
import operator as op
from functools import reduce


def double_factorial(n):
    """
    Return the double factorial of a number
    """

    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)


def comb(n, r):
    """
    Return the combination of n elements in r sets
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def theo_win_wolf(n_players):
    """
    Return the theoretical win prob for wolves given the env
    """
    m = math.floor(math.sqrt(n_players))
    n = n_players

    theoretical_p_win = 1

    for i in range(m + 1):
        num = comb(m, i) * double_factorial(n - i)
        num *= (-1) ** i
        den = double_factorial(n) * double_factorial((n % 2) - i)
        nd = num / den
        theoretical_p_win -= nd

    return theoretical_p_win


def save_results(rows: list, f_name):
    """
    Save the result of a tuning experiment in csv format
    """
    # the headers
    headers = ['num players', 'mean']

    rows.insert(0, headers)

    with open(f_name, "w") as file:
        wr = csv.writer(file)
        wr.writerows(rows)


def theo_ww_revenge(n):
    try:
        return 1 - 1 / math.pow(n, math.floor(math.sqrt(n)) + 1)
    except ZeroDivisionError:
        return 1


def theo_unite():
    return 1


if __name__ == '__main__':
    rg = 15.0
    num_p = list(range(5, 100))
    # v_win=[theo_win_wolf(i) for i in num_p]
    v_win = [theo_ww_revenge(i) for i in num_p]
    # v_win = [theo_unite() for i in num_p]
    save_results(list(zip(num_p, v_win)), "theo_ww_revenge.csv")
