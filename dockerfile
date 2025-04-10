FROM python:3

WORKDIR /usr/src/app

# install uv
RUN pip install --no-cache-dir uv

# copy over pyproject for building project dependencies
COPY pyproject.toml .

# install python libraries needed
RUN uv pip install --no-cache-dir -r pyproject.toml --system

# copy over project
COPY ./src ./src
COPY ./data/raw ./data/raw
COPY ./data/processed/.gitkeep ./data/processed/.gitkeep

# setup debugging
RUN uv pip install --no-cache-dir debugpy --system

# expose debugging port
EXPOSE 5678 

# run debug commands
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "./src/stage_data_sql.py"]

# run commands
# CMD [ "python", "./src/stage_data_sql.py", "./src/spc_solver.py" ] 