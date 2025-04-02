###############
# <Libraries> #
###############

from typing import Literal, Self

import polars as pl

################
# <\Libraries> #
################
###############
# <SPC Class> #
###############


class SpcSolver:
    sample_size: int
    time_frame: str
    time_series: pl.Series
    data_series: pl.Series

    def __init__(
        self: Self, time_frame: str = "d", sample_size: int = 30
    ) -> None:
        self.sample_size = sample_size
        self.time_frame = time_frame

    def solve(
        self: Self,
        time_series: pl.Series,
        data_series: pl.Series,
    ) -> Self:
        # TODO(Tom Babb): write docstring

        # check time & data_series have compatible shape
        if time_series.shape[0] != data_series.shape[0]:
            msg = (
                "time_series and data_series "
                "do not have the same number of rows!"
            )
            raise ValueError(msg)

        # check time & data series have > sample size number of samples
        if time_series.shape[0] < self.sample_size:
            msg = "Too few datapoints. Need more datapoints than the sample size."
            raise ValueError(msg)

        # create dataframe for calculations
        data_df = pl.DataFrame(data={'date': time_series, 'data': data_series},schema=pl.Schema({'date': pl.Date(),'data': pl.Float64()}))

        # index for spc fit number
        data_df=data_df.with_columns(pl.when(pl.arange(0,pl.count()).lt(self.sample_size)).then(0).otherwise(None).alias('index'))

        # calculate initial regression fit
        beta_0: float = pl.cov(data_series[0 : self.sample_size],time_series[0 : self.sample_size]) / time_series[0 : self.sample_size].var()
        alpha_0: float = data_series[0 : self.sample_size].mean() - beta_0 * time_series[0 : self.sample_size].mean()

        # calculate initial residuals
        residuals_0: pl.Series = data_series - alpha_0 + beta_0 * time_series

        # find Z values
        z_values_0: pl.Series = (residuals_0 - residuals_0.mean() )/ residuals_0.std()

        # find outliers
        outliers_0: pl.Series = z_values_0.abs() >= 3

        # find first outlier
        outliers_0.first()


        return self


################
# <\SPC Class> #
################
