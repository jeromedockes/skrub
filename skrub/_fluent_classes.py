import inspect


def _to_arg(name):
    assert name.endswith("_")
    return name.removesuffix("_")


def _setattr_default(obj, attr_name, value):
    if hasattr(obj, attr_name):
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
    cls.__init__ = _make_init(args, kwargs)
    cls.__repr__ = _repr
    _setattr_default(cls, "_copy", _copy)
    _setattr_default(cls, "_get_options_repr", _get_options_repr)
    _setattr_default(cls, "_get_pos_args_repr", _get_pos_args_repr)
    _setattr_default(cls, "_get_factory_repr", _get_factory_repr)
    _setattr_default(cls, "_to_dict", _to_dict)
    _setattr_default(cls, "_with_params", _with_params)
    for arg in [*args, *kwargs]:
        _setattr_default(cls, arg, _make_setter(arg))
    return cls


def _get_options_repr(self):
    parts = []
    attr_dict = self._to_dict()
    for k, v in self._kw_fields.items():
        attr = attr_dict[k]
        if attr != v:
            parts.append(f"{k}({attr!r})")
    if not parts:
        return ""
    return "." + ".".join(parts)


def _make_func(lines):
    exec("".join(lines), globals(), d := {})
    return d.popitem()[1]


def _make_init(args, kwargs):
    sig = ", ".join(["self", *args, *(f"{k}={v!r}" for k, v in kwargs.items())])
    lines = (
        [f"def __init__({sig}):\n"]
        + [f"    self.{name}_ = {name}\n" for name in [*args, *kwargs]]
        + [
            "    if hasattr(self, '__post_init__'):\n",
            "        self.__post_init__()\n",
        ]
    )
    return _make_func(lines)


def _copy(self):
    return self.__class__(**self._to_dict())


def _make_setter(arg):
    lines = [
        f"def {arg}(self, new):\n",
        f"    return self._with_params({arg}=new)\n",
    ]
    return _make_func(lines)


def _to_dict(self):
    d = {}
    for k in self.__class__._fields:
        d[k] = getattr(self, k + "_")
    return d


def _with_params(self, **new_params):
    return self.__class__(**(self._to_dict() | new_params))


def _get_pos_args_repr(self):
    attr_dict = self._to_dict()
    return ", ".join(repr(attr_dict[k]) for k in self._pos_fields)


def _get_factory_repr(self):
    args = _get_pos_args_repr(self)
    return f"{self.__class__.__name__}({args})"


def _repr(self):
    opts = self._get_options_repr()
    return f"{self._get_factory_repr()}{opts}"
