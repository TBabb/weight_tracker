###############
# <Libraries> #
###############

from typing import Self

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
    data_df: pl.DataFrame

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
        data_df = pl.DataFrame(data={"date": time_series, "data": data_series})

        # ensure "date" column has date dtype, if is string dtype, then convert
        # TODO(Tom Babb):  this shouldnt be necessary
        # - typing is going wrong when querying from database, need to fix
        if data_df["date"].dtype == pl.String():
            data_df = data_df.with_columns(
                pl.col("date")
                .str.to_datetime(format=r"%d/%m/%Y", strict=True, exact=True)
                .alias("date")
            )

        # index for spc fit number
        index_no: int = 0

        # create integer date column
        data_df = data_df.with_columns(
            (pl.col("date") - pl.col("date").min())
            .dt.total_days()
            .alias("date_int")
        )

        # create null columns for later use
        data_df = data_df.with_columns(
            pl.lit(None).alias("alpha"),
            pl.lit(None).alias("beta"),
            pl.lit(None).alias("regression_value"),
            pl.lit(None).alias("residual"),
            pl.lit(None).alias("residual_mean"),
            pl.lit(None).alias("residual_std"),
            pl.lit(None).alias("z_score"),
            pl.lit(None).alias("outlier_bool"),
        )

        date_start: int = 0
        regression_limit: int = date_start + self.sample_size - 1

        while date_start < data_df.shape[0] - 1:
            data_df = data_df.with_columns(
                pl.when(pl.col("date_int").ge(pl.lit(date_start)))
                .then(index_no)
                .otherwise(pl.col("date_int"))
                .alias("index")
            )
            index_no += 1

            training_interval_expr: pl.Expr = (
                pl.col("date_int")
                .le(regression_limit)
                .and_(pl.col("date_int").ge(date_start))
            )

            beta: float = data_df.filter(training_interval_expr).select(
                pl.cov(
                    pl.col("data"),
                    pl.col("date_int"),
                )
                / pl.col("date_int").var()
            )[0, 0]

            alpha: float = data_df.filter(training_interval_expr).select(
                pl.col("data").mean() - beta * pl.col("date_int").mean()
            )[0, 0]

            data_df = (
                data_df.with_columns(
                    pl.when(pl.col("date_int").ge(date_start))
                    .then(beta)
                    .otherwise(pl.col("beta"))
                    .alias("beta"),
                    pl.when(pl.col("date_int").ge(date_start))
                    .then(alpha)
                    .otherwise(pl.col("alpha"))
                    .alias("alpha"),
                )
                .with_columns(
                    (alpha + beta * pl.col("date_int")).alias(
                        "regression_value"
                    )
                )
                .with_columns(
                    (pl.col("data") - pl.col("regression_value")).alias(
                        "residual"
                    )
                )
            )

            residual_mean: float = data_df.filter(
                training_interval_expr
            ).select(pl.col("residual").mean())[0, 0]

            residual_std: float = data_df.filter(
                training_interval_expr
            ).select(pl.col("residual").std())[0, 0]

            data_df = (
                data_df.with_columns(
                    pl.when(pl.col("date_int").ge(date_start))
                    .then(pl.lit(residual_mean))
                    .otherwise(pl.col("residual_mean"))
                    .alias("residual_mean"),
                    pl.when(pl.col("date_int").ge(date_start))
                    .then(pl.lit(residual_std))
                    .otherwise(pl.col("residual_std"))
                    .alias("residual_std"),
                )
                .with_columns(
                    pl.when(pl.col("date_int").ge(date_start))
                    .then(
                        (pl.col("residual") - pl.col("residual_mean"))
                        / pl.col("residual_std")
                    )
                    .otherwise(pl.col("z_score"))
                    .alias("z_score")
                )
                .with_columns(
                    (pl.col("z_score").abs() >= 3).alias("outlier_bool")
                )
            )

            first_outlier: int = data_df.filter(
                pl.col("outlier_bool").and_(
                    pl.col("date_int").ge(pl.lit(date_start))
                )
            ).filter(pl.col("date_int").eq(pl.col("date_int").min()))[
                "date_int"
            ][0]

            date_start = first_outlier - 1
            regression_limit = date_start + self.sample_size - 1

            self.data_df = data_df

        return self


################
# <\SPC Class> #
################
