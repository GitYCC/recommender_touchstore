from datetime import datetime

import pandas as pd

from process import split_data_by_datetime


def _dt2ts(dt):
    timestamp = int((dt - datetime(1970, 1, 1)).total_seconds())
    return timestamp


def test_split_data_by_datetime():
    df = pd.DataFrame({'dummy': ['A', 'B', 'C'],
                       'timestamp': list(map(_dt2ts,
                                             [datetime(2000, 1, 1),
                                              datetime(2005, 1, 1),
                                              datetime(2010, 1, 1)]))})
    cut_dt = datetime(2005, 1, 1)
    expect_df_before = pd.DataFrame({'dummy': ['A', ],
                                     'timestamp': list(map(_dt2ts, [datetime(2000, 1, 1), ]))})
    expect_df_after = pd.DataFrame({'dummy': ['B', 'C', ],
                                    'timestamp': list(map(_dt2ts, [datetime(2005, 1, 1),
                                                                   datetime(2010, 1, 1), ]))})
    df_before, df_after = split_data_by_datetime(df, cut_dt)
    pd.testing.assert_frame_equal(df_before, expect_df_before)
    pd.testing.assert_frame_equal(df_after, expect_df_after)