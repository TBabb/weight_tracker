###############
# <Libraries> #
###############

from typing import Self

import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    _spc_intervals_df: pl.DataFrame

    def __init__(
        self: Self,
        time_frame: str = "1d",
        sample_size: int = 30,
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

        data_df = data_df.group_by_dynamic("date", every=self.time_frame).agg(
            pl.col("data").count().alias("count_data"),
            pl.col("data").mean().alias("mean_data"),
            pl.col("data").std().alias("std_data"),
        )

        # index for spc fit number
        spc_index_no: int = 0

        # create integer date column
        data_df = data_df.with_columns(
            pl.col("date").cum_count().alias("index")
        )

        # create null columns for later use
        data_df = data_df.with_columns(
            pl.lit(None).alias("spc_index"),
            pl.lit(None).alias("mean_alpha"),
            pl.lit(None).alias("mean_beta"),
            pl.lit(None).alias("regression_value"),
            pl.lit(None).alias("residual"),
            pl.lit(None).alias("residual_mean"),
            pl.lit(None).alias("residual_std"),
            pl.lit(None).alias("z_score"),
            pl.lit(None).alias("outlier_bool"),
        )

        date_start: int = 0
        regression_limit: int = date_start + self.sample_size - 1
        max_index: int = data_df.select(pl.col("index").max())[0, 0]

        while date_start < max_index:
            data_df = data_df.with_columns(
                pl.when(pl.col("index").ge(pl.lit(date_start)))
                .then(spc_index_no)
                .otherwise(pl.col("spc_index"))
                .alias("spc_index")
            )
            spc_index_no += 1

            training_interval_expr: pl.Expr = (
                pl.col("index")
                .le(regression_limit)
                .and_(pl.col("index").ge(date_start))
            )

            mean_beta: float = data_df.filter(training_interval_expr).select(
                pl.cov(
                    pl.col("mean_data"),
                    pl.col("index"),
                )
                / pl.col("index").var()
            )[0, 0]

            mean_alpha: float = data_df.filter(training_interval_expr).select(
                pl.col("mean_data").mean() - mean_beta * pl.col("index").mean()
            )[0, 0]

            data_df = (
                data_df.with_columns(
                    pl.when(pl.col("index").ge(date_start))
                    .then(mean_beta)
                    .otherwise(pl.col("mean_beta"))
                    .alias("mean_beta"),
                    pl.when(pl.col("index").ge(date_start))
                    .then(mean_alpha)
                    .otherwise(pl.col("mean_alpha"))
                    .alias("mean_alpha"),
                )
                .with_columns(
                    (
                        pl.col("mean_alpha")
                        + pl.col("mean_beta") * pl.col("index")
                    ).alias("regression_value")
                )
                .with_columns(
                    (pl.col("mean_data") - pl.col("regression_value")).alias(
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
                    pl.when(pl.col("index").ge(date_start))
                    .then(pl.lit(residual_mean))
                    .otherwise(pl.col("residual_mean"))
                    .alias("residual_mean"),
                    pl.when(pl.col("index").ge(date_start))
                    .then(pl.lit(residual_std))
                    .otherwise(pl.col("residual_std"))
                    .alias("residual_std"),
                )
                .with_columns(
                    pl.when(pl.col("index").ge(date_start))
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

            # check if any outliers,
            # if not we're done, if so create next regression
            if (
                data_df.filter(pl.col("index").gt(date_start))[
                    "outlier_bool"
                ].sum()
                > 0
            ):
                first_outlier: int = data_df.filter(
                    pl.col("outlier_bool").and_(
                        pl.col("index").ge(pl.lit(date_start))
                    )
                ).filter(pl.col("index").eq(pl.col("index").min()))["index"][0]

                date_start = first_outlier + 1
                regression_limit = date_start + self.sample_size - 1
                print(date_start)
            else:
                break

        spc_intervals_data_df: pl.DataFrame = (
            data_df.group_by(
                [
                    "spc_index",
                    "mean_alpha",
                    "mean_beta",
                    "residual_mean",
                    "residual_std",
                ]
            )
            .agg(
                pl.col("spc_index").count().alias("num_data_points"),
                pl.col("date").min().alias("start_date"),
                pl.col("date").max().alias("end_date"),
            )
            .sort("spc_index")
        )

        self._spc_intervals_df = spc_intervals_data_df

        self.data_df = data_df

        return self

    def plot(
        self: Self,
        min_date: pl.Date | None = None,
        max_date: pl.Date | None = None,
    ) -> tuple[Figure, list[Axes]]:
        fig = plt.figure()

        # mean data plot
        ax_1 = fig.add_subplot(2, 1, 1)

        plt.xlabel("Date")
        plt.ylabel("Mean Data")
        plt.title("Mean Data SPC Plot")

        plt.minorticks_on()
        plt.grid(
            visible=True,
            which="minor",
            axis="both",
            linestyle=":",
            color="k",
            linewidth=1,
        )
        plt.grid(
            visible=True,
            which="major",
            axis="both",
            linestyle="-",
            color="k",
            linewidth=1,
        )

        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"],
            "k:",
            linewidth=2,
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"] + self.data_df["residual_std"],
            "k:",
            linewidth=2,
            label="fit $\\pm$ 2 s.t.d",
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"] - self.data_df["residual_std"],
            "k:",
            linewidth=2,
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"]
            + 2 * self.data_df["residual_std"],
            "k:",
            linewidth=2,
            label="fit $\\pm$ 2 s.t.d",
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"]
            - 2 * self.data_df["residual_std"],
            "k:",
            linewidth=2,
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"]
            + 3 * self.data_df["residual_std"],
            "k:",
            linewidth=2,
            label="fit $\\pm$ 3 s.t.d",
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["regression_value"]
            - 3 * self.data_df["residual_std"],
            "k:",
            linewidth=2,
        )
        plt.plot(
            self.data_df["date"],
            self.data_df["mean_data"],
            "o",
            markerfacecolor="r",
            markeredgecolor="k",
            markersize=8,
            markeredgewidth=1,
            label="mean data",
        )

        if min_date is not None:
            plt.xlim(left=min_date)
        if max_date is not None:
            plt.xlim(right=max_date)

        plt.legend(loc="upper right")

        # standard deviation plot
        ax_2 = fig.add_subplot(2, 1, 2)

        plt.xlabel("Date")
        plt.ylabel("$\\sigma$ Data")
        plt.title("$\\sigma$ Data SPC Plot")

        plt.minorticks_on()
        plt.grid(
            visible=True,
            which="minor",
            axis="both",
            linestyle=":",
            color="k",
            linewidth=1,
        )
        plt.grid(
            visible=True,
            which="major",
            axis="both",
            linestyle="-",
            color="k",
            linewidth=1,
        )

        plt.plot(
            self.data_df["date"],
            self.data_df["std_data"],
            "k:",
            linewidth=2,
        )

        if min_date is not None:
            plt.xlim(left=min_date)
        if max_date is not None:
            plt.xlim(right=max_date)

        plt.legend(loc="upper right")

        return fig, [ax_1, ax_2]


################
# <\SPC Class> #
################
