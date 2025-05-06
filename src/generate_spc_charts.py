# TODO(Tom Babb): write better docstring

###############
# <Libraries> #
###############

import sqlite3
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt

import spc_solver as spc

################
# <\Libraries> #
################
###########
# <Paths> #
###########

input_sqlite_path: Path = (
    Path().cwd() / "data" / "processed" / "sql_database.db"
)

############
# <\Paths> #
############
###################
# <Main Function> #
###################


def main(input_sqlite_path: Path | str = input_sqlite_path) -> None:
    #########################
    # <Connect to Database> #
    #########################

    conn: sqlite3.Connection = sqlite3.connect(input_sqlite_path)

    ##########################
    # <\Connect to Database> #
    ##########################
    #############################
    # <Read Data From Database> #
    #############################

    query: str = "SELECT * FROM staging_mass_data"

    staging_data_schema = pl.Schema(
        {
            "date": pl.String(),
            "mass_kg": pl.Float64(),
            "muscle_mass_kg": pl.Float64(),
            "fat_mass_kg": pl.Float64(),
            "fat_free_mass_kg": pl.Float64(),
            "skeletal_mass_percent": pl.Float64(),
            "subcutaneous_fat_mass_percent": pl.Float64(),
            "visceral_fat_number": pl.Float64(),
            "water_mass_percent": pl.Float64(),
            "protein_mass_percent": pl.Float64(),
        }
    )

    staging_data: pl.DataFrame = pl.read_database(
        connection=conn,
        query=query,
        schema_overrides=staging_data_schema,
    )

    staging_data = staging_data.with_columns(
        pl.col("date").cast(pl.Date()).alias("date")
    )

    ##############################
    # <\Read Data From Database> #
    ##############################
    ###########################
    # <Run SPC Mass Analysis> #
    ###########################

    daily_mass_spc: spc.SpcSolver = spc.SpcSolver().solve(
        staging_data["date"], staging_data["mass_kg"]
    )

    weekly_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1w").solve(
        staging_data["date"],
        staging_data["mass_kg"],
    )

    monthly_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1mo").solve(
        staging_data["date"],
        staging_data["mass_kg"],
    )

    quarterly_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1q").solve(
        staging_data["date"],
        staging_data["mass_kg"],
    )

    annual_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1y").solve(
        staging_data["date"],
        staging_data["mass_kg"],
    )

    ############################
    # <\Run SPC Mass Analysis> #
    ############################
    ##########################
    # <Run SPC Fat Analysis> #
    ##########################

    daily_fat_mass_spc: spc.SpcSolver = spc.SpcSolver().solve(
        staging_data["date"], staging_data["fat_mass_percent"]
    )

    weekly_fat_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1w").solve(
        staging_data["date"],
        staging_data["fat_mass_percent"],
    )

    monthly_fat_mass_spc: spc.SpcSolver = spc.SpcSolver(
        time_frame="1mo"
    ).solve(
        staging_data["date"],
        staging_data["fat_mass_percent"],
    )

    quarterly_fat_mass_spc: spc.SpcSolver = spc.SpcSolver(
        time_frame="1q"
    ).solve(
        staging_data["date"],
        staging_data["fat_mass_percent"],
    )

    annual_fat_mass_spc: spc.SpcSolver = spc.SpcSolver(time_frame="1y").solve(
        staging_data["date"],
        staging_data["fat_mass_percent"],
    )

    ###########################
    # <\Run SPC Fat Analysis> #
    ###########################
    ###########
    # <Plot> #
    ###########

    mass_fig_1, mass_ax_1 = daily_mass_spc.plot()

    mass_fig_2, mass_ax_2 = weekly_mass_spc.plot()

    mass_fig_3, mass_ax_3 = monthly_mass_spc.plot()

    mass_fig_4, mass_ax_4 = quarterly_mass_spc.plot()

    mass_fig_5, mass_ax_5 = annual_mass_spc.plot()

    fat_mass_fig_1, fat_mass_ax_1 = daily_fat_mass_spc.plot()

    fat_mass_fig_2, fat_mass_ax_2 = weekly_fat_mass_spc.plot()

    fat_mass_fig_3, fat_mass_ax_3 = monthly_fat_mass_spc.plot()

    fat_mass_fig_4, fat_mass_ax_4 = quarterly_fat_mass_spc.plot()

    fat_mass_fig_5, fat_mass_ax_5 = annual_fat_mass_spc.plot()

    plt.show()

    ###########
    # <\Plot> #
    ###########

    pass


####################
# <\Main Function> #
####################

if __name__ == "__main__":
    main(input_sqlite_path=input_sqlite_path)
