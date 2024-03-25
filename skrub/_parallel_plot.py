import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["get_parallel_coord_data", "plot_parallel_coord", "DEFAULT_COLORSCALE"]
DEFAULT_COLORSCALE = "bluered"


def plot_parallel_coord(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("please install plotly.")
        return None
    return go.Figure(
        data=go.Parcoords(
            **get_parallel_coord_data(
                cv_results,
                metadata,
                colorscale=colorscale,
            )
        )
    )


def get_parallel_coord_data(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    prepared_columns = [
        _prepare_column(cv_results[col_name], col_name in metadata["log_scale_columns"])
        for col_name in cv_results.columns
    ]
    return dict(
        line=dict(
            color=cv_results["mean_score"],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=dict(text="mean_score")),
        ),
        dimensions=prepared_columns,
    )


def _prepare_column(col, is_log_scale):
    if not pd.api.types.is_numeric_dtype(col):
        return _prepare_obj_column(col)
    if is_log_scale:
        return _prepare_log_scale_column(col)
    return {"label": col.name, "values": col.to_numpy()}


def _prepare_obj_column(col):
    encoder = OrdinalEncoder()
    encoded_col = encoder.fit_transform(pd.DataFrame({col.name: col})).ravel()
    return {
        "label": col.name,
        "values": encoded_col,
        "tickvals": np.arange(len(encoder.categories_[0])),
        "ticktext": encoder.categories_[0],
    }


def _prepare_log_scale_column(col):
    vals = np.log(col.to_numpy())
    min_val, max_val = vals.min(), vals.max()
    tickvals = np.linspace(min_val, max_val, 10)
    if pd.api.types.is_integer_dtype(col):
        ticktext = [str(int(np.round(np.exp(v)))) for v in tickvals]
    else:
        ticktext = list(map("{:.2g}".format, np.exp(tickvals)))
    return {
        "label": col.name,
        "values": vals,
        "tickvals": tickvals,
        "ticktext": ticktext,
    }
