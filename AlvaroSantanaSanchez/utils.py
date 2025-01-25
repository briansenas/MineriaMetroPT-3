# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.predict(X[:2, :])
# def manual_fold_training(
#         clf_model : LogisticRegression,
#         data : pl.DataFrame,
#         fold_number : int = 0,
#         *,
#         target_column : str = "is_anomaly"
# ):
#     scores = {"fold":fold_number}
#
#     fold_train = data.filter(pl.col("fold") != fold_number).drop("fold")
#     fold_test = data.filter(pl.col("fold") == fold_number).drop("fold")
#     X_train = (fold_train.select(pl.exclude(target_column))
#                .select(pl.col(pl.Float64)).with_columns(
#         pl.all().fill_nan(0)
#     ).to_numpy())
#     y_train =  fold_train.select(target_column).to_series().to_list()
#     clf_model.fit(X_train,y_train)
#     X_test = fold_test.select(pl.exclude(target_column)).to_numpy()
#     y_test =  fold_test.select(pl.col(target_column)).to_series().to_numpy()
#     y_pred = clf_model.predict(X_test)
#     scores["f1_score"] = f1_score(y_test,y_pred)
#     scores["recall"] = recall_score(y_test,y_pred)
#
#     return scores
#
#
# logistic_regressor = LogisticRegression(solver='liblinear',C=100.0,penalty="l2",max_iter=1000)
#
# fold_scores = []
# for fold in train["fold"].unique().to_list():
#     fold_scores.append(manual_fold_training(clf_model=logistic_regressor,
#                                             data=train,
#                                             fold_number=fold))
#
#
# df_scores = pl.DataFrame(fold_scores)
# mean_row = df_scores.select(pl.all().mean().cast(pl.Float64))
# df_with_mean = df_scores.with_columns(pl.col("fold").cast(pl.Utf8)).vstack(mean_row)
# df_with_mean
