import datetime
import html
import io
import re
import shutil
import traceback
import webbrowser
from pathlib import Path

import jinja2
from sklearn.base import BaseEstimator

from .. import _dataframe as sbd
from .. import datasets
from .._reporting import TableReport
from .._reporting._serve import open_in_browser
from .._utils import Repr, random_string
from ._choosing import Choice
from ._evaluation import choices, clear_results, evaluate, graph, param_grid
from ._expressions import Apply, Value, Var
from ._utils import simple_repr

# TODO after merging the expressions do some refactoring and move this stuff to
# _reporting. better to do it later to avoid conflicts with independent changes
# to the _reporting module.


def _get_jinja_env():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            Path(__file__).resolve().parents[1]
            / "_reporting"
            / "_data"
            / "templates"
            / "expressions",
            encoding="UTF-8",
        ),
        autoescape=True,
    )
    return env


def _get_template(template_name):
    return _get_jinja_env().get_template(template_name)


def node_report(expr, mode="preview", environment=None, **report_kwargs):
    result = evaluate(expr, mode=mode, environment=environment)
    if sbd.is_column(result):
        # TODO say in page that it was a column not df
        # maybe this should be handled by tablereport? or we should have a
        # seriesreport with just 1 card?
        result_df = sbd.make_dataframe_like(result, [result])
        result_df = sbd.copy_index(result, result_df)
        result = result_df
    if sbd.is_dataframe(result):
        report = TableReport(result, **report_kwargs)
    else:
        try:
            report = result._repr_html_()
        except Exception:
            res_repr = Repr()
            res_repr.maxstring = 1000
            res_repr.maxother = 10_000
            report = _get_template("simple-repr.html").render(
                {"object_repr": res_repr.repr(result)}
            )
    return report


def _get_output_dir(output_dir, overwrite):
    now = datetime.datetime.now().isoformat()
    if output_dir is None:
        output_dir = (
            datasets.get_data_dir()
            / "execution_reports"
            / f"full_expr_report_{now}_{random_string()}"
        )
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"The output directory already exists: {output_dir}. "
                    "Set 'overwrite=True' to allow replacing it."
                )
    output_dir.mkdir(parents=True)
    return output_dir


def _node_status(expr_graph, mode):
    status = {}
    for node_id, node in expr_graph["nodes"].items():
        if mode in node._skrub_impl.results:
            status[node_id] = "success"
        elif mode in node._skrub_impl.errors:
            status[node_id] = "error"
        else:
            status[node_id] = "none"
    return status


def full_report(
    expr,
    environment=None,
    mode="preview",
    clear=True,
    open=True,
    output_dir=None,
    overwrite=False,
):
    if clear:
        clear_results(expr, mode)
    try:
        return _do_full_report(
            expr,
            environment=environment,
            mode=mode,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    finally:
        if clear:
            clear_results(expr, mode)


def _do_full_report(
    expr,
    environment=None,
    mode="preview",
    open=True,
    output_dir=None,
    overwrite=False,
):
    output_dir = _get_output_dir(output_dir, overwrite)
    try:
        # TODO dump report in callback instead of evaluating full expression
        # first, so that we can clear intermediate results
        result = evaluate(expr, mode=mode, environment=environment, clear=False)
        evaluate_error = None
    except Exception as e:
        result = None
        evaluate_error = e
    g = graph(expr)
    node_status = _node_status(g, mode)
    node_rindex = {id(node): k for k, node in g["nodes"].items()}

    def node_name_to_url(node_name):
        return f"node_{node_name}.html"

    def make_url(node):
        return node_name_to_url(node_rindex[id(node)])

    svg = draw_expr_graph(expr, url=make_url).decode("utf-8")
    jinja_env = _get_jinja_env()
    index = jinja_env.get_template("index.html").render(
        {"svg": svg, "node_status": node_status}
    )
    index_file = output_dir / "index.html"
    index_file.write_text(index, "utf-8")

    for i, node in g["nodes"].items():
        report, error, error_msg = None, None, None
        if mode in node._skrub_impl.results:
            report = node_report(node, mode=mode, environment=environment)
        elif mode in node._skrub_impl.errors:
            e = node._skrub_impl.errors[mode]
            error = "".join(traceback.format_exception(e))
            error_msg = "".join(traceback.format_exception_only(e))
            if hasattr(e, "__notes__"):
                error_msg = error_msg.removesuffix("\n".join(e.__notes__) + "\n")
        if isinstance(report, TableReport):
            print(f"Generating report for node {i}")
            report = report.html_snippet()
        node_parents = g["parents"].get(i, [])
        node_parents = [
            {
                "id": n,
                "description": simple_repr(g["nodes"][n]),
                "url": node_name_to_url(n),
            }
            for n in g["parents"].get(i, [])
        ]
        node_children = [
            {
                "id": n,
                "description": simple_repr(g["nodes"][n]),
                "url": node_name_to_url(n),
            }
            for n in g["children"].get(i, [])
        ]
        if isinstance(node._skrub_impl, Apply):
            estimator = getattr(
                node._skrub_impl, "estimator_", node._skrub_impl.estimator
            )
            if isinstance(estimator, BaseEstimator):
                estimator_html_repr = estimator._repr_html_()
            else:
                estimator_html_repr = None
        else:
            estimator_html_repr = None
        node_page = jinja_env.get_template("node.html").render(
            dict(
                total_n_nodes=len(g["nodes"]),
                node_nb=i,
                node_parents=node_parents,
                node_children=node_children,
                node_repr=simple_repr(node),
                report=report,
                error=error,
                error_msg=error_msg,
                node_creation_stack_description=node._skrub_impl.creation_stack_description(),
                node_description=node._skrub_impl.description,
                node_name=node._skrub_impl.name,
                node_type=node._skrub_impl.__class__.__name__,
                svg=svg,
                node_status=node_status,
                estimator_html_repr=estimator_html_repr,
            )
        )
        out = output_dir / f"node_{i}.html"
        out.write_text(node_page, "utf-8")

    index_file = index_file.resolve()
    output = {"result": result, "error": evaluate_error, "report_path": index_file}
    if not open:
        return output
    webbrowser.open(f"file://{index_file}")
    return output


class _SVG(bytes):
    def _repr_html_(self):
        return self.decode("utf-8")

    def open(self):
        open_in_browser(
            _get_template("graph.html").render({"svg": self.decode("utf-8")})
        )


def _add_style(kwargs, *new_styles):
    if "style" in kwargs:
        style = [s.strip() for s in kwargs["style"].split(",")]
    else:
        style = []
    style = ",".join([*style, *new_styles])
    kwargs["style"] = style


def _node_kwargs(expr, url=None):
    label = html.escape(simple_repr(expr))
    if (
        not isinstance(expr._skrub_impl, Var)
        and (name := expr._skrub_impl.name) is not None
    ):
        label = f"{label}\n{name!r}"
    kwargs = {
        "shape": "box",
        "fontsize": 12,
        "height": 0.3,
        "margin": "0.1,0.08",
        "labelloc": "c",
        "fontname": "sans-serif",
        "color": "black",
    }
    if expr._skrub_impl.is_X:
        label = f"X: {label}"
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#c6d5f0"
    elif expr._skrub_impl.is_y:
        label = f"y: {label}"
        kwargs["style"] = "filled"
        kwargs["fillcolor"] = "#fad9c6"
    if url is not None and (computed_url := url(expr)) is not None:
        kwargs["URL"] = computed_url
        label = label.replace("\n", "<br />")
        label = f'<<FONT COLOR="#1a0dab"><B>{label}</B></FONT>>'
    kwargs["label"] = label
    tooltip = html.escape(expr._skrub_impl.creation_stack_last_line())
    if description := expr._skrub_impl.description:
        tooltip = f"{tooltip}\n\n{html.escape(description)}"
    kwargs["tooltip"] = tooltip
    if isinstance(expr._skrub_impl, (Var, Value)):
        kwargs["peripheries"] = 2
    return kwargs


def _dot_id(n):
    return f"node_{n}"


def draw_expr_graph(expr, url=None, direction="TB"):
    import pydot

    g = graph(expr)
    dot_graph = pydot.Dot(rankdir=direction)
    for node_id, e in g["nodes"].items():
        kwargs = _node_kwargs(e, url=url)
        kwargs["id"] = _dot_id(node_id)
        node = pydot.Node(_dot_id(node_id), **kwargs)
        dot_graph.add_node(node)
    for c, parents in g["parents"].items():
        for p in parents:
            dot_graph.add_edge(pydot.Edge(_dot_id(p), _dot_id(c)))

    svg = dot_graph.create_svg()
    svg = re.sub(b"<title>.*?</title>", b"", svg)
    return _SVG(svg)


def describe_param_grid(expr):
    grid = param_grid(expr)
    expr_choices = choices(expr)

    buf = io.StringIO()
    for subgrid in grid:
        prefix = "- "
        for k, v in subgrid.items():
            choice = expr_choices[k]
            name = choice.name or repr(choice)
            if isinstance(choice, Choice) and isinstance(v, list):
                if len(v) == 1:
                    v = choice.outcomes[v[0]]
                else:
                    v = f'[{", ".join(map(str, (choice.outcomes[idx] for idx in v)))}]'
                buf.write(f"{prefix}{name}: {v}\n")
            else:
                buf.write(f"{prefix}{name}: {v}\n")
            prefix = "  "
    return buf.getvalue() or "<empty parameter grid>"
