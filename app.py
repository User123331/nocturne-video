"""Nocturne Video entry point. Pass --open to open the dashboard in your browser."""
import subprocess
import sys

from local_app.server import main

if __name__ == "__main__":
    main(open_browser="--open" in sys.argv)
