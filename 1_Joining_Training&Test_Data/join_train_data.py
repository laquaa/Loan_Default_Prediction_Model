import polars as pl
from set_table_dtypes import set_table_dtypes

def replace_values(value):
    mappings = {
        'SINGLE': 0.0,
        'MARRIED': 1.0,
        'DIVORCED': 2.0,
        'LIVING_WITH_PARTNER':3.0,
        'WIDOWED':4.0,
        'FALSE': 0.0,
        'TRUE': 1.0,
        'SALARIED_GOVT': 0.0,
        'EMPLOYED': 1.0,
        'PRIVATE_SECTOR_EMPLOYEE': 2.0,
        'RETIRED_PENSIONER': 3.0,
        'SELFEMPLOYED': 4.0,
        'OTHER': 5.0,
        'HANDICAPPED_2': 6.0,
        'HANDICAPPED_3': 7.0
    }
    return mappings.get(value, value)

train_static_cb = pl.read_csv("../csv_files/train/train_static_cb_0.csv").pipe(set_table_dtypes)
train_static_cb = (
    train_static_cb
    .with_columns([
        (pl.col('days90_310L') - pl.col('days30_165L')).alias('days30-90'),
        (pl.col('days120_123L') - pl.col('days90_310L')).alias('days90-120'),
        (pl.col('days180_256L') - pl.col('days120_123L')).alias('days120-180'),
        (pl.col('days360_512L') - pl.col('days180_256L')).alias('days180-360')
    ])
    .select(['case_id', 'days30_165L', 'days30-90', 'days90-120', 'days120-180', 'days180-360'])
)
train_static = pl.concat(
    [
        pl.read_csv("../csv_files/train/train_static_0_0.csv").pipe(set_table_dtypes),
        pl.read_csv("../csv_files/train/train_static_0_1.csv").pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
train_static = train_static.with_columns(
    pl.when(pl.col("currdebt_22A").is_null())
    .then(pl.col("currdebtcredtyperange_828A"))
    .when(pl.col("currdebtcredtyperange_828A").is_null())
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") == 0) & (pl.col("currdebt_22A") != 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") != 0) & (pl.col("currdebt_22A") == 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") != 0) & (pl.col("currdebt_22A") != 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") == 0) & (pl.col("currdebt_22A") == 0))
    .then(0)
    .otherwise(None)
    .alias("current_debt")
)
columns_of_interest = ['case_id','actualdpdtolerance_344P','amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'credamount_770A', 'disbursedcredamount_1113A', 'eir_270L', 'currdebt_22A', 'currdebtcredtyperange_828A','current_debt','totalsettled_863A','pmtnum_254L']
train_static = train_static[columns_of_interest]
train_person_1 = (
    pl.read_csv("../csv_files/train/train_person_1.csv")
    .pipe(set_table_dtypes)
    .select(['case_id', 'familystate_447L', 'incometype_1044T', 'mainoccupationinc_384A', 'safeguarantyflag_411L', 'num_group1'])
    .with_columns(pl.col('incometype_1044T').map_elements(replace_values).cast(pl.Float64))
    .with_columns(pl.col('familystate_447L').map_elements(replace_values).cast(pl.Float64))
    .with_columns(pl.col('safeguarantyflag_411L').map_elements(replace_values).cast(pl.Float64))
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_applprev_1 = pl.concat(
    [
        pl.read_csv("../csv_files/train/train_applprev_1_0.csv").pipe(set_table_dtypes),
        pl.read_csv("../csv_files/train/train_applprev_1_1.csv").pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
approved_counts = (
    train_applprev_1
    .filter(pl.col('status_219L') == 'A')
    .group_by('case_id')
    .agg(pl.count('status_219L').alias('approved_applications'))
)
train_applprev_1 = train_applprev_1.join(approved_counts, on='case_id', how='left')
train_applprev_1 = (
    train_applprev_1
    .filter(pl.col('num_group1') == 0)
    .select(['case_id', 'mainoccupationinc_437A', 'approved_applications'])
    .with_columns(
        pl.col('approved_applications').fill_null(0)
    )
)
train_other_1 = pl.read_csv("../csv_files/train/train_other_1.csv")
train_other_1 = (
    train_other_1
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_deposit_1 = pl.read_csv("../csv_files/train/train_deposit_1.csv")
deposit_sums = (
    train_deposit_1
    .group_by('case_id')
    .agg(pl.sum('amount_416A').alias('deposit'))
)
train_deposit_1 = train_deposit_1.join(deposit_sums, on='case_id', how='left')
train_deposit_1 = (
    train_deposit_1
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
    .drop('contractenddate_991D')
    .drop('openingdate_313D')
    .drop('amount_416A')
)
train_credit_bureau_a_1 = pl.concat(
    [
        pl.read_csv("../csv_files/train/train_credit_bureau_a_1_0.csv").pipe(set_table_dtypes),
        pl.read_csv("../csv_files/train/train_credit_bureau_a_1_1.csv").pipe(set_table_dtypes),
        pl.read_csv("../csv_files/train/train_credit_bureau_a_1_2.csv").pipe(set_table_dtypes),
        pl.read_csv("../csv_files/train/train_credit_bureau_a_1_3.csv").pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
train_credit_bureau_a_1 = (
    train_credit_bureau_a_1
    .select(['case_id', 'debtoutstand_525A', 'debtoverdue_47A', 'num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)

train_basetable = pl.read_csv("../csv_files/train/train_base.csv")
def process_week_num(week_num):
    return week_num % 52 if week_num >= 52 else week_num
train_basetable = train_basetable.with_columns(
    pl.col("WEEK_NUM").map_elements(process_week_num).alias("WEEK_NUM")
)
train_basetable = (
    train_basetable
    .join(train_static, on='case_id', how='left')
    .join(train_static_cb, on='case_id', how='left')
    .join(train_person_1, on='case_id', how='left')
    .join(train_applprev_1, on='case_id', how='left')
    .join(train_deposit_1, on='case_id', how='left')
    .join(train_other_1, on='case_id', how='left')
    .join(train_credit_bureau_a_1, on='case_id', how='left')
    .drop('date_decision')
    .drop('MONTH')
)

train_basetable_special_case = train_basetable.filter(pl.col('eir_270L').is_null())
train_basetable_base_case = train_basetable.filter(pl.col('eir_270L').is_not_null())

train_basetable_base_case = train_basetable_base_case.sort("case_id")
train_basetable_special_case = train_basetable_special_case.sort("case_id")
train_basetable_special_case = train_basetable_special_case.to_pandas()
train_basetable_base_case = train_basetable_base_case.to_pandas()
train_basetable_base_case.to_csv("../csv_files/train_basetable_merged.csv",index=False)
train_basetable_special_case.to_csv("../csv_files/train_basetable_merged_special_case.csv",index=False)