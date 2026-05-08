from grid_model import create_sample_grid
from layout_validator import print_validation_report, validate_layout


def main() -> None:
    grid = create_sample_grid()
    result = validate_layout(grid)
    print_validation_report(result)


if __name__ == "__main__":
    main()
