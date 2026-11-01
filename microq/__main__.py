"""Entry point: ``python -m microq ...`` (see microq.cli)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
