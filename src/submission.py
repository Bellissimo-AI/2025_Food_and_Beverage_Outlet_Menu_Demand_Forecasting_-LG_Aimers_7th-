
import pandas as pd

def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    pred_df = pred_df.copy()
    pred_df['영업장명_메뉴명'] = pred_df['영업장명_메뉴명'].apply(lambda x: x[0] if isinstance(x, tuple) else x)

    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))

    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)

    return final_df
