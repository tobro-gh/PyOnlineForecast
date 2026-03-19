# Quickstart

Get started with PyOnlineForecast.

## 1) Prerequisites

- Python 3.12.3+
- Git

## 2) Clone repository

```bash
git clone https://github.com/tobro-gh/PyOnlineForecast
cd PyOnlineForecast
```

## 3) Install options
Pick one of the options below,
### Standard install

```bash
python -m pip install .
```
Use if you do not want the install to depend on `pandas`.

### Full install with `forecast_tools` using `pandas`

```bash
python -m pip install ".[forecast_tools]"
```
Use if you would like to use `forecast_tools` e.g. for running examples.

### Development install (editable)

```bash
python -m pip install -e ".[forecast_tools]"
```

Use editable mode if you plan to modify package code while testing.

## 4) Run notebooks

Open and run:

- `docs/examples/basics.ipynb`
- `docs/examples/forecast_example.ipynb`
- `docs/examples/hierarchy_example.ipynb`

> Note: `docs/examples/forecast_example.ipynb` uses `scipy`.  
> Install it manually if needed:
>
> ```bash
> python -m pip install scipy
> ```