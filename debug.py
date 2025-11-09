import lightgbm as lgb
from lightgbm import early_stopping

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=100,
    callbacks=[early_stopping(stopping_rounds=10)],
    verbose_eval=False,
)
