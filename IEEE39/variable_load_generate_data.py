import sys
from textwrap import dedent


def main():
    message = dedent(
        """
        variable_load_generate_data.py has been replaced by ambient mode in IEEE39/generate_data.py.

        Use:
          python IEEE39/generate_data.py --ambient

        Examples:
          python IEEE39/generate_data.py --ambient
          python IEEE39/generate_data.py --ambient --scenario ambient_test
          python IEEE39/generate_data.py --ambient --duration 900 --ambient-magnitude-percent 0.2
        """
    ).strip()
    print(message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
