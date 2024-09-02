# Building

## Build

```console
python -m pip install -U build
python -m build
```

## Publish

_Get token from https://pypi.org/manage/account/token_

```console
python -m pip install -U twine

python -m build
python -m twine upload --repository testpypi dist/*
```

_Remove `--repository testpypi` to upload to real repository_

## Test

```console
pip install .[dev]
pytest -s .
```