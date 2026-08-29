import pytest

from tippytop.llm.semantic_validation import validate_prediction_paths


def test_prediction_path_rejects_training_only_columns() -> None:
    source = '''
class Model:
    def transform(self, rows):
        return rows[["is_click", "long_view"]]

    def predict(self, rows):
        return self.transform(rows)

def fit(train_rows, seed):
    return Model()

def predict(model, rows):
    return model.predict(rows)
'''

    with pytest.raises(ValueError, match="is_click.*long_view"):
        validate_prediction_paths(source)


def test_prediction_path_allows_persisted_training_aggregates() -> None:
    source = '''
class Model:
    def fit_maps(self, rows):
        self.click_map = rows.groupby("video_id")["is_click"].mean().to_dict()

    def transform_prediction(self, rows):
        return rows["video_id"].map(self.click_map).fillna(0.0)

    def predict(self, rows):
        return self.transform_prediction(rows)

def fit(train_rows, seed):
    model = Model()
    model.fit_maps(train_rows)
    return model

def predict(model, rows):
    return model.predict(rows)
'''

    validate_prediction_paths(source)


def test_unrelated_training_function_may_read_outcomes() -> None:
    source = '''
def training_features(rows):
    return rows["play_time_ms"]

def fit(train_rows, seed):
    return {"mean": float(training_features(train_rows).mean())}

def predict(model, rows):
    return rows["duration_ms"].to_numpy() * 0.0 + model["mean"]
'''

    validate_prediction_paths(source)
