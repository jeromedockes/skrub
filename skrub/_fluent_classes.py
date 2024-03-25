import inspect


def _to_arg(name):
    if not name.endswith("_"):
        raise ValueError("Fluent class attribute names must end with '_'")
    return name.removesuffix("_")


def _to_attribute(name):
    return name + "_"


def _setattr_default(obj, attr_name, value):
    if attr_name in obj.__dict__:
        return
    setattr(obj, attr_name, value)


def fluent_class(cls):
    annotations = inspect.get_annotations(cls)
    args, kwargs = [], {}
    for anno in annotations:
        if hasattr(cls, anno):
            kwargs[_to_arg(anno)] = getattr(cls, anno)
        else:
            args.append(_to_arg(anno))
    cls._fields = (*args, *kwargs)
    cls._pos_fields = args
    cls._kw_fields = kwargs
    _setattr_default(cls, "__init__", _make_init(args, kwargs))
    _setattr_default(cls, "__repr__", _repr)
    _setattr_default(cls, "_to_dict", _to_dict)
    _setattr_default(cls, "_with_params", _with_attributes)
    _setattr_default(cls, "_get_setters_snippet", _get_setters_snippet)
    for arg in cls._fields:
        _setattr_default(cls, arg, _make_setter(arg))
    return cls


def _get_setters_snippet(self):
    parts = []
    attr_dict = self._to_dict()
    for k, v in self._kw_fields.items():
        attr = attr_dict[k]
        if attr != v:
            parts.append(f"{k}({attr!r})")
    if not parts:
        return ""
    return "." + ".".join(parts)


def _make_func(lines, local_vars={}):
    exec("".join(lines), globals(), d := dict(local_vars))
    return d.popitem()[1]


def _make_init(args, kwargs):
    sig = ", ".join(["self", *args, *(f"{k}={k}" for k in kwargs)])
    lines = (
        [f"def __init__({sig}):\n"]
        + [f"    self.{name}_ = {name}\n" for name in [*args, *kwargs]]
        + [
            "    if hasattr(self, '__post_init__'):\n",
            "        self.__post_init__()\n",
        ]
    )
    return _make_func(lines, kwargs)


def _make_setter(arg):
    lines = [
        f"def {arg}(self, new):\n",
        f"    return self._with_params({arg}=new)\n",
    ]
    return _make_func(lines)


def _to_dict(self):
    return {k: getattr(self, _to_attribute(k)) for k in self.__class__._fields}


def _with_attributes(self, **new_attributes):
    return self.__class__(**(self._to_dict() | new_attributes))


def _repr(self):
    args_repr = ", ".join(
        [repr(getattr(self, _to_attribute(a))) for a in self._pos_fields]
        + [
            f"{k}={v!r}"
            for k, d in self._kw_fields.items()
            if (v := getattr(self, _to_attribute(k))) != d
        ]
    )
    return f"{self.__class__.__name__}({args_repr})"
