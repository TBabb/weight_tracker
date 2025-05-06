"""Script to import data from csv into a sqlite database."""
# TODO(Tom Babb): write better docstring

###############
# <Libraries> #
###############

import sqlite3
from pathlib import Path

import polars as pl
from sqlalchemy import Engine, create_engine

################
# <\Libraries> #
################
###########
# <Paths> #
###########

input_csv_path = Path().cwd() / "data" / "raw" / "full_data.csv"
output_sqlite_path = Path().cwd() / "data" / "processed" / "sql_database.db"

############
# <\Paths> #
############
###################
# <Main Function> #
###################


def main(
    input_csv_path: Path | str = input_csv_path,
    output_sqlite_path: Path | str = output_sqlite_path,
) -> None:
    """Stage csv data into sqlite database.

    Parameters
    ----------
    input_csv_path: pathlib.Path | str
        Path to csv that we want to stage.
    output_sqlite_path: pathlib.Path | str
        Path to sqlite database.
    """
    ##############
    # <Read csv> #
    ##############

    staging_data_schema: pl.Schema = pl.Schema(
        {
            "date": pl.Date(),
            "mass_kg": pl.Float64(),
            "muscle_mass_kg": pl.Float64(),
            "fat_mass_percent": pl.Float64(),
            "bone_mass_kg": pl.Float64(),
            "fat_free_mass_kg": pl.Float64(),
            "skeletal_muscle_mass_percent": pl.Float64(),
            "subcutaneous_fat_mass_percent": pl.Float64(),
            "visceral_fat_no": pl.Float64(),
            "water_mass_percent": pl.Float64(),
            "protein_mass_percent": pl.Float64(),
        }
    )

    try:
        staging_data_df: pl.DataFrame = pl.read_csv(
            input_csv_path, schema_overrides=staging_data_schema
        )
    except Exception:
        raise

    ###############
    # <\Read csv> #
    ###############
    #########################
    # <Connect to Database> #
    #########################

    # create database if it does not already exist
    sqlite3.connect(output_sqlite_path)

    # create sqlalchemy connection to database for manipulation
    if isinstance(output_sqlite_path, str):
        conn_uri: str = "sqlite:///" + output_sqlite_path
    else:
        conn_uri = "sqlite:///" + str(output_sqlite_path.resolve())
    sql_engine: Engine = create_engine(conn_uri)

    ##########################
    # <\Connect to Database> #
    ##########################
    ##########################
    # <Create Staging Table> #
    ##########################

    # create / update staging table
    with sql_engine.connect() as conn:
        staging_data_df.write_database(
            table_name="staging_mass_data",
            connection=conn,
            if_table_exists="replace",
        )

    ###########################
    # <\Create Staging Table> #
    ###########################

    temp = 1


####################
# <\Main Function> #
####################

if __name__ == "__main__":
    main(input_csv_path=input_csv_path, output_sqlite_path=output_sqlite_path)
