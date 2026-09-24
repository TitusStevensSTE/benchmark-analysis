"""Run the canonical GLUE Phase 1 LODO analysis."""

from current_simple_crossbenchmark import run_glue


def main() -> None:
    run_glue()
    print("\nCanonical GLUE outputs: results/phase1_crossbenchmark/glue")


if __name__ == "__main__":
    main()
