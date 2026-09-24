"""Run the canonical ImageNet Phase 1 LODO analysis."""

from current_simple_crossbenchmark import run_imagenet


def main() -> None:
    run_imagenet()
    print("\nCanonical ImageNet outputs: results/phase1_crossbenchmark/imagenet")


if __name__ == "__main__":
    main()
