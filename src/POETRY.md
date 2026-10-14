# Poetry or plain Python

The project uses **Poetry**, but runs just as well with **plain Python and pip**: both install the same versions and produce byte-identical submission files.

**Poetry** is a dependency manager. It declares the packages in `pyproject.toml`, locks their exact versions in `poetry.lock`, and installs them in an isolated virtual environment. `requirements.txt` mirrors the lock for pip users. Never edit it by hand.

Python ≥ 3.14 (the locked packages support 3.12+). The `report` group (reportlab) is only needed for `src/report.py`.

## Install

```bash
# Poetry (install it once with: pipx install poetry)
poetry install --no-root --with report

# or plain Python
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Same commands, from the repository root. **With Poetry, prefix them with `poetry run`**, or activate its environment once with `eval $(poetry env activate)`.

```bash
python -m src.data_preparation   # then src.train, src.tuning, src.test_evaluation, src.predict, src.report
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

## Change dependencies

```bash
poetry add <package>                  # or: poetry add --group report <package> / poetry remove <package>
python -m src.sync_requirements          # regenerate requirements.txt from poetry.lock
python -m src.sync_requirements --check  # verify it is up to date
```
