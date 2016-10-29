# -*- encoding:utf-8
import unittest

import pandas as pd
import numpy as np
from qn import qn


class TestQN(unittest.TestCase):
    def test_simple(self):
        """
        Checking with the simple table filled with already sorted values in each column.
        As a result we should get the table with the same values in each row.
        """

        table = pd.DataFrame({'A': range(10), 'B': range(11,21), 'C': range(132,142), 'D': range(1000, 1010)})
        tmp = np.mean(np.array(table.T), axis = 1)[np.array([[i]*table.shape[0] for i in range(table.shape[1])])]

        # what should we get
        table_res_ref = pd.DataFrame(tmp.T, index = table.index, columns=table.columns)

        # what we get
        table_res_test = qn(table)
        self.assertTrue(table_res_test.equals(table_res_ref))

    def test_real(self):
        filename = 'test_data/test_data.csv'
        df = pd.read_csv(filename)
        self._check_sanity(qn(df))

    def _check_sanity(self, df):
        """
        checking if everything is ok
        with the values in sorted rows.
        they must be the same.
        df -- is a resulted df from qn-algo

        // if the rows of df.T will
        // be sorted in each column
        // values in each row must be the same.

        If everything is ok, 0 will be returned.
        Else -- return the number of rows where there're different values.

        """

        srtd, idx = _sort_columns_individually(df.T)

        maxs = np.max(srtd, axis=1)
        mins = np.min(srtd, axis=1)
        rows_with_different_values = sum([True for i in range(len(mins)) if mins[i] != maxs[i]])
        self.assertEquals(rows_with_different_values, 0)


def _sort_columns_individually(df):
    df_srtd = pd.concat(
            [df[col].sort_values().reset_index(drop=True) for col in df],
            axis=1, ignore_index=True)
    indexes = pd.concat(
            [df[col].argsort() for col in df],
            axis=1, ignore_index=True)
    return df_srtd, indexes


if __name__ == '__main__':
    unittest.main()

