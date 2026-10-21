"""Enable `python -m faircode`."""

from .cli import main

# Covered by test_python_dash_m_faircode_runs_the_cli (#966), which runs it in a subprocess that
# coverage does not measure, hence the pragma (#986).
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
