# CircleCI Setup and Workflow Documentation

This document describes the CI/CD pipeline for the RL Intraday Futures Strategies project using CircleCI.

## Overview
CircleCI automates testing, linting, and training smoke tests for every commit and pull request. The configuration is defined in `.circleci/config.yml`.

## Workflow Steps
1. **install-deps**: Installs Python dependencies in a virtual environment using `requirements.txt`.
2. **lint**: Runs code linting with Ruff on the `src` and `tests` folders.
3. **test**: Runs unit tests using Pytest.
4. **train-smoke**: Runs a short training loop to verify the RL pipeline works end-to-end.

### Job Dependencies
- `lint` and `test` require `install-deps`.
- `train-smoke` requires `test`.

## How to Use
- Push changes to GitHub; CircleCI will automatically trigger the workflow.
- Check the CircleCI dashboard for build status and logs.
- To validate the config locally, install the CircleCI CLI and run:
  ```bash
  circleci config validate
  ```

## Troubleshooting
- If a job fails, check the logs for missing dependencies or test errors.
- Ensure all Python scripts and configs are up to date.

## File Locations
- `.circleci/config.yml`: Main CircleCI configuration file.
- `requirements.txt`: Python dependencies.
- `src/`, `tests/`: Source code and tests.

## Notes
- The pipeline is designed for production and will catch most errors before deployment.
- For custom jobs, edit `.circleci/config.yml` and commit changes.

---
For more details, see the [CircleCI documentation](https://circleci.com/docs/).
