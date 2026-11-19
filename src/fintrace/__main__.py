"""Entry point for ``python -m fintrace`` (Docker ENTRYPOINT). CLI is not implemented yet."""

__all__ = ["main"]


def main() -> int:
    """Return the process exit code.

    TODO: dispatch subcommands (demo, baselines, attacks, oracle-gap).
    """
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
