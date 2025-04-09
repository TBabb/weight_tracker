FROM python:3

WORKDIR /usr/src/app

COPY pyproject.toml ./

RUN pip install uv
RUN uv pip install --no-cache-dir -r pyproject.toml --system

COPY . .

CMD [ "python", "./src/stage_data_sql.py", "./src/spc_solver.py" ] 