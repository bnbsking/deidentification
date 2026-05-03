#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/app

poetry run pip install -e . --no-deps
## building docker step has already installed all 3rd-party packages (exclude self package: deid)
## init step install the self package: deid only (without internet, with editable mode)

# poetry run pip install -e .[full]

#poetry run python /app/scripts/databases/build.py  # build if not exist

# poetry run python /app/scripts/main/clear_tmp.py  # clear tmp files

poetry run uvicorn deid.serve:app --reload --host 0.0.0.0 --port 8006

tail -f /dev/null
