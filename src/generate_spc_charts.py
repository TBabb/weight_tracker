# TODO(Tom Babb): write better docstring

###############
# <Libraries> #
###############

import sqlite3
from pathlib import Path

import polars as pl

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

    mass_spc: spc.SpcSolver = spc.SpcSolver().solve(
        staging_data["date"], staging_data["mass_kg"]
    )

    #######################
    # <\Run SPC Analysis> #
    #######################
    ###########
    # <Plot> #
    ###########

    fig, ax = mass_spc.plot()

    ###########
    # <\Plot> #
    ###########

    pass


####################
# <\Main Function> #
####################

if __name__ == "__main__":
    main(input_sqlite_path=input_sqlite_path)
