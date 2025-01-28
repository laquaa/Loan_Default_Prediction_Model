
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample



train_data = pd.read_csv('train_basetable_merged.csv', index_col='case_id')
models = {}


def assessment_no_stacked(csv_name):
    case_ids = []
    all_predictions = []
    df = pd.read_csv(csv_name, index_col='case_id')

    for i in range(len(df)):

        sample = df.iloc[i]

        train_data_0 = train_data.drop(sample.name)

        case_ids.append(sample.name)
        sample = sample.dropna()
        non_null_columns = sample.index.tolist()
        sample_dict = {col: [val] for col, val in zip(non_null_columns, sample.values)}
        sample = pd.DataFrame(sample_dict)
        sample.columns = sample.columns.map(str)
        sample = sample.astype('float32')
        columns_key = '_'.join(non_null_columns)

        if columns_key not in models:

            non_null_columns_target = non_null_columns.copy()
            non_null_columns_target.append('target')
            chosen = train_data_0[non_null_columns_target]
            chosen = chosen.dropna()

            majority_class = chosen[chosen.target == 0]
            minority_class = chosen[chosen.target == 1]

            if len(minority_class) <= 3 or len(majority_class) <= 3:
                all_predictions.append(0)

            else:
                majority_downsampled = resample(majority_class,
                                                replace=False,
                                                n_samples=len(minority_class) * 7,
                                                random_state=10)
                chosen = pd.concat([majority_downsampled, minority_class])

                X = chosen.drop('target', axis=1)
                X.columns = X.columns.map(str)
                X = X.astype('float32')
                y = chosen['target']

                model = LogisticRegression(random_state=10, n_jobs=-1)

                model.fit(X, y)
                result = model.predict_proba(sample)
                all_predictions.append(result[:, 1][0])
                models[columns_key] = model

        else:
            result = models[columns_key].predict_proba(sample)
            all_predictions.append(result[:, 1][0])

    output = {
        'case_id': case_ids,
        'score': all_predictions
    }
    df = pd.DataFrame(output)

    return df



import logging
logging.getLogger().setLevel(logging.ERROR)

from sklearnex import patch_sklearn
patch_sklearn()


results = assessment_no_stacked('a.csv')
results.set_index('case_id', inplace=True)
sorted_df = results.sort_index()
sorted_df.to_csv('submission_LGBM_only.csv')



from sklearn.metrics import roc_auc_score

predict = sorted_df['score'].values
true = pd.read_csv("b.csv")

auc = roc_auc_score(true, predict)
print(auc)
