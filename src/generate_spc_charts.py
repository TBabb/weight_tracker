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

    staging_data: pl.DataFrame = pl.read_database(
        connection=conn, query=query, schema_overrides=pl.Schema()
    )

    cur = conn.cursor()

    data = cur.execute(query)

    ##############################
    # <\Read Data From Database> #
    ##############################
    ######################
    # <Run SPC Analysis> #
    ######################

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

    #######################
    # <\Run SPC Analysis> #
    #######################
    ###########
    # <Plot> #
    ###########

    fig, ax = daily_mass_spc.plot()

    fig, ax = weekly_mass_spc.plot()

    fig, ax = monthly_mass_spc.plot()

    fig, ax = quarterly_mass_spc.plot()

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
