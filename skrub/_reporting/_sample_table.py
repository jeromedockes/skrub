import pandas as pd

# def pick_row_counts(df, max_head_rows, max_tail_rows):
#     # return n_head_rows, n_tail_rows, is_ellided


class _LevelCounter:
    def __init__(self, level):
        self.level = level
        self.current_heading = None

    def start_new_cell(self):
        self.current_heading = None

    def get_cell(self, index_entry):
        value = index_entry[self.level]
        if self.current_heading is not None and self.current_heading["value"] == value:
            self.current_heading["span"] += 1
            return None
        self.current_heading = {"value": value, "span": 1}
        return self.current_heading


def _multi_index_headings(idx, orientation):
    counters = [_LevelCounter(i) for i in range(len(idx.levels))]
    rows = []
    for idx_entry in idx.to_flat_index():
        new_row = []
        rows.append(new_row)
        counters[-1].start_new_cell()
        for i in range(len(counters)):
            c = counters[i]
            cell = c.get_cell(idx_entry)
            if cell is not None:
                for child_c in counters[i + 1 :]:
                    child_c.start_new_cell()
            new_row.append(cell)

    if orientation == "horizontal":
        rows = list(zip(*rows))
        span_name = "col_span"
    else:
        assert orientation == "vertical"
        span_name = "row_span"
    for r in rows:
        for cell in r:
            if cell is not None:
                cell[span_name] = cell.pop("span")
    return rows


def _to_multi(pd_index):
    if isinstance(pd_index, pd.MultiIndex):
        return pd_index
    return pd.MultiIndex.from_arrays(
        [pd_index], names=[n] if (n := pd_index.name) is not None else None
    )


def _n_levels(pd_index):
    if (levels := getattr(pd_index, "levels", None)) is not None:
        return len(levels)
    return 1


def _level_names(pd_index):
    if (names := getattr(pd_index, "names", None)) is not None:
        return names
    return [None] * _n_levels(pd_index)


def make_table(df, max_head_rows=5, max_tail_rows=5):
    parts = []

    min_i = -_n_levels(df.columns)
    min_j = -_n_levels(df.index)

    n_df_rows = df.shape[0]
    if n_df_rows <= max_head_rows + max_tail_rows:
        n_head_rows, n_tail_rows, is_ellided = n_df_rows, 0, False
    else:
        n_head_rows, n_tail_rows, is_ellided = max_head_rows, max_tail_rows, True

    def add_header():
        column_headings = _multi_index_headings(
            _to_multi(df.columns), orientation="horizontal"
        )
        header = {"name": "header", "elem": "thead", "rows": []}
        parts.append(header)
        for i, level_name, row in zip(
            range(min_i, 0), df.columns.names, column_headings
        ):
            header_row = [
                {
                    "col_span": -min_j,
                    "i": i,
                    "j": min_j,
                    "value": level_name,
                    "elem": "th",
                }
            ]
            for j, c in enumerate(row):
                if c is not None:
                    c["i"] = i
                    c["j"] = j
                    c["elem"] = "th"
                    c["scope"] = "row"
                    c["column_index"] = j
                    if i == -1:
                        c["row_span"] = 2
                    header_row.append(c)
            header["rows"].append(header_row)

        col_names_row = []
        header["rows"].append(col_names_row)
        for j, level_name in zip(range(min_j, 0), _level_names(df.index)):
            col_names_row.append(
                {"j": j, "i": -1, "value": level_name, "elem": "th", "scope": "column"}
            )

    def add_body(sub_df, name, start_row):
        body = {"name": name, "elem": "tbody", "rows": []}
        parts.append(body)
        index_headings = _multi_index_headings(
            _to_multi(sub_df.index), orientation="vertical"
        )
        for i_offset, (heading_row, df_row) in enumerate(
            zip(index_headings, sub_df.itertuples(index=False))
        ):
            i = start_row + i_offset
            body_row = []
            for j, h in zip(range(min_j, 0), heading_row):
                if h is not None:
                    h["j"] = j
                    h["i"] = i
                    h["elem"] = "th"
                    h["scope"] = "row"
                    body_row.append(h)
            body_row += [
                {"value": v, "i": i, "j": j, "elem": "td", "column_idx": j}
                for j, v in enumerate(df_row)
                if v is not None
            ]
            body["rows"].append(body_row)

    def add_ellipsis():
        parts.append({"name": "ellipsis", "elem": "tbody"})

    add_header()
    add_body(df.iloc[:n_head_rows], "head", 0)
    if is_ellided:
        add_ellipsis()
    if n_tail_rows:
        add_body(df.iloc[-n_tail_rows:], "tail", n_head_rows)

    return {"parts": parts, "min_j": min_j, "max_j": df.shape[1]}


if __name__ == "__main__":
    import pandas as pd

    data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    cols = pd.MultiIndex(
        levels=[["ca"], ["cca", "ccb"]], codes=[[0, 0], [0, 1]], names=["c0", "c1"]
    )
    rows = pd.MultiIndex(
        levels=[["ra", "rb"], ["rra", "rrb"], ["rrra", "rrrb"]],
        codes=[[0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]],
        names=["r0", "r1", "r2"],
    )

    df = pd.DataFrame(data, columns=cols, index=rows)

    from skrub import datasets

    employees = datasets.fetch_employee_salaries().X
    # TableReport(df).open()

    df = employees
    table = make_table(df)
    print(df)
    import pprint

    pprint.pprint(table)
    print(df)
    from skrub import _dataframe as sbd

    print(sbd.column_names(df))
