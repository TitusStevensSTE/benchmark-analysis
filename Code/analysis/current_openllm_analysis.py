"""Run the canonical OpenLLM Phase 1 LODO analysis."""

from current_simple_crossbenchmark import run_openllm


def main() -> None:
    run_openllm()
    print("\nCanonical OpenLLM outputs: results/phase1_crossbenchmark/openllm")


if __name__ == "__main__":
    main()
