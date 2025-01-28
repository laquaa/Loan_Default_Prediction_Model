import polars as pl
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if col[-1] == "D":
            df = df.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(col))
        elif col[-1] in ["M", "T"]:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
        elif col[-1] in ["P", "A"]:
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        elif col[-1] == "L":
            non_null_values = df.filter(pl.col(col).is_not_null())[col].limit(1).to_list()
            if non_null_values:
                first_non_null = non_null_values[0]
                if isinstance(first_non_null, bool):
                    df = df.with_columns(pl.col(col).cast(pl.Boolean).alias(col))
                elif isinstance(first_non_null, (float, int)):
                    df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                else:
                    df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
    return df