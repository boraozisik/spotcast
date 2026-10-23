import sys
import pandas as pd


def main() -> None:
    print(f"Python sürümü: {sys.version.split()[0]}")
    print(f"pandas sürümü: {pd.__version__}")

    spotlar = pd.DataFrame(
        {
            "spot": ["Kuşadası Pamucak", "Alaçatı", "Akyaka"],
            "spor": ["kano", "windsurf", "kitesurf"],
            "ruzgar_knot": [8, 18, 22],
        }
    )
    print(spotlar)
    print('******************')
    print(spotlar["ruzgar_knot"].mean())
    print('******************')
    print(spotlar[spotlar["ruzgar_knot"] > 10])
    print('******************')
    print(spotlar.sort_values("ruzgar_knot", ascending=False))


if __name__ == "__main__":
    main()