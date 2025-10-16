# ejecuta: python -m optimization.remodel.tests.smoke_test
from optimization.remodel.config import PATHS
from optimization.remodel.io import load_base_df

if __name__ == "__main__":
    df = load_base_df()
    print("filas:", len(df))
    print(df.columns.tolist()[:20], "...")