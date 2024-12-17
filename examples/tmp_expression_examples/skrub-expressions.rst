Skrub expressions
-----------------

Skrub expressions are a way to build complex pipelines with arbitrary inputs.

An expression is a combination of constants, names, operators or function calls
that can be evaluated to produce a value.

For example here is a python expression represented as a string:

>>> e = '2 * 3 + a'

We can evaluate it, providing bindings for the variables it contains:

>>> eval(e, {'a': 10})
16
>>> eval(e, {'a': 100})
106

Skrub expressions are similar, but

- they are constructed from python objects rather than being parsed from strings
- they can involve scikit-learn estimators. This makes it possible to use them
  to build machine-learning pipelines

The main way to create a skrub expression is to create a variable with
``skrub.var()`` and call some methods or apply some operators to it.

>>> import skrub

Here we create a variable named 'a', representing an input to our
machine-learning pipeline.

>>> e1 = skrub.var('a')
>>> e1
<Var 'a'>

We can construct other expressions by applying operators to it

>>> e2 = 3 + e1
>>> e2
<BinOp: add>

We can evaluate an expression, providing bindings for the variables it contains:

>>> e2.skb.get_value({'a': 10})
13

>>> e2.skb.get_value({'a': 100})
103

We can also inspect the computation steps required to evaluate an expression

>>> print(e2.skb.describe_steps())
VAR 'a'
BINOP: add

Or visually:

>>> e2.skb.draw_graph().open()

Adding estimators
-----------------

Of course we are not really interested in applying operators to ints but rather to dataframes

>>> e3 = skrub.var('df')
>>> e3
<Var 'df'>
>>> e4 = e3.drop(columns='noise').assign(width_cm=e3['width_m'] * 10)

>>> import pandas as pd
>>> df = pd.DataFrame({'ID': [1, 2, 3], 'width_m': [0.5, 1.5, 2.5], 'noise': '_'})
>>> df
   ID  width_m noise
0   1      0.5     _
1   2      1.5     _
2   3      2.5     _
>>> e4.skb.get_value({'df': df})
   ID  width_m  width_cm
0   1      0.5       5.0
1   2      1.5      15.0
2   3      2.5      25.0

>>> df_1 = pd.DataFrame({'ID': [2, 2], 'width_m': [3.5, 4.5], 'noise': '_'})
>>> e4.skb.get_value({'df': df_1})
   ID  width_m  width_cm
0   2      3.5      35.0
1   2      4.5      45.0

So far we could have done this with a simple python function. But computation
steps in a skrub expression can also fit and remember relevant state, which
makes it possible to use them to build stateful machine learning pipelines.

>>> from sklearn.preprocessing import OneHotEncoder
>>> e5 = e4.skb.apply(OneHotEncoder(sparse_output=False), cols='ID')

To get a fittable pipeline, we call ``.skb.get_estimator()``

>>> estimator = e5.skb.get_estimator()
>>> estimator.fit_transform({'df': df})
   width_m  width_cm  ID_1  ID_2  ID_3
0      0.5       5.0   1.0   0.0   0.0
1      1.5      15.0   0.0   1.0   0.0
2      2.5      25.0   0.0   0.0   1.0
>>> estimator.transform({'df': df_1})
   width_m  width_cm  ID_1  ID_2  ID_3
0      3.5      35.0   0.0   1.0   0.0
1      4.5      45.0   0.0   1.0   0.0

Note that the items used by the one-hot encoder have been retained between the
call to ``fit_transform`` and to ``transform``.

``skrub.deferred``
------------------

We have seen how to create a variable with ``skrub.var``, and that we can apply
operators to it and call its methods.

Sometimes we may need to apply a function to a variable in our pipeline.

>>> def expand(df, added):
...     assert isinstance(df, pd.DataFrame), f'not a dataframe: {type(df)}'
...     return pd.concat([df, df + added], ignore_index=True)

Naturally, we cannot apply this function directly to our expression

>>> expand(e5, 17)
Traceback (most recent call last):
    ...
AssertionError: not a dataframe: <class 'skrub._expressions._expressions.Expr'>

So we can use ``skrub.deferred``. It delays the application of our function.
Rather than applying it immediately, it returns another skrub expression with an
added step. The function will be called when we evaluate the expression, i.e.
when we run our pipeline.

>>> skrub.deferred(expand)
<function expand at 0x74927fd5c720>
>>> e6 = skrub.deferred(expand)(e5, 17)
>>> print(e6.skb.describe_steps())
VAR 'df'
CALLMETHOD 'drop'
( VAR 'df' )*
GETITEM 'width_m'
BINOP: mul
CALLMETHOD 'assign'
APPLY OneHotEncoder
CALL 'expand'
* Cached, not recomputed

We can see that rather than being called immediately, a "CALL 'expand'" step has
been added to the pipeline.

>>> e6.skb.get_value({'df': df})
   width_m  width_cm  ID_1  ID_2  ID_3
0      0.5       5.0   1.0   0.0   0.0
1      1.5      15.0   0.0   1.0   0.0
2      2.5      25.0   0.0   0.0   1.0
3     17.5      22.0  18.0  17.0  17.0
4     18.5      32.0  17.0  18.0  17.0
5     19.5      42.0  17.0  17.0  18.0

Preview data
------------

So far we have been building our expressions with abstract variables for which
we only provide bindings when we run the pipeline.

This can make debugging hard because we only see errors at the end, when we
evaluate the whole expression. Skrub allows us to provide placeholders, preview
data for our variables. When we provide such data, the expressions are evaluated
as soon as possible on the preview data. This allows us to inspect the results
of intermediate steps, and to trigger errors sooner.

Here the second argument to ``skrub.var`` is the preview data

>>> e = skrub.var('df', pd.DataFrame({'width_m': [2.3, 4.5], 'ID': [1, 2], 'noise': '?'}))
>>> e
<Var 'df'>
Preview:
――――――――
       width_m  ID noise
    0      2.3   1     ?
    1      4.5   2     ?

>>> e = e.drop(columns='noise').assign(width_mm=e['width_m'] * 1000)
>>> print(e.skb.describe_steps())
VAR 'df'
CALLMETHOD 'drop'
( VAR 'df' )*
GETITEM 'width_m'
BINOP: mul
CALLMETHOD 'assign'
* Cached, not recomputed
>>> e
<CallMethod 'assign'>
Preview:
――――――――
       width_m  ID  width_mm
    0      2.3   1    2300.0
    1      4.5   2    4500.0

If we want to get the preview result as a python object and manipulate it we can
access the ``.preview`` property:

>>> e.skb.preview
   width_m  ID  width_mm
0      2.3   1    2300.0
1      4.5   2    4500.0

>>> type(e)
<class 'skrub._expressions._expressions.Expr'>
>>> type(e.skb.preview)
<class 'pandas.core.frame.DataFrame'>

Hyperparameters
---------------

Expressions can be parametrized by using ``skrub.choose_from``, ``skrub.choose_int``
or ``skrub.choose_float``.
We can then ask skrub to build a hyperparameter grid and tune the choices to
produce the best prediction performance on the training data.

>>> to_add = skrub.choose_from([100, 2000], name='ID offset')
>>> e = e.assign(ID=e['ID'] + to_add)
>>> e = e.drop(columns=skrub.choose_from(['width_m', 'width_mm'], name='dropped cols'))
>>> e
<CallMethod 'drop'>
Preview:
――――――――
        ID  width_mm
    0  101    2300.0
    1  102    4500.0

>>> print(e.skb.describe_param_grid())
- ID offset: [100, 2000]
  dropped cols: ['width_m', 'width_mm']

Cross-validation
----------------

To cross-validate pipelines and tune parameters we need to tell skrub what
constitute our features ``X`` and target ``y``, which define the cross-validation
splits.

We can do this with ``mark_as_x()`` and ``mark_as_y()``

>>> from sklearn.datasets import fetch_openml
>>> import skrub

>>> features, target = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

>>> X = skrub.var('X', features).skb.mark_as_x()
>>> y = skrub.var('y', target).skb.mark_as_y()
>>> X
<Var 'X'>
Preview:
――――――――
          pclass  ...                        home.dest
    0          1  ...                     St Louis, MO
    1          1  ...  Montreal, PQ / Chesterville, ON
    2          1  ...  Montreal, PQ / Chesterville, ON
    3          1  ...  Montreal, PQ / Chesterville, ON
    4          1  ...  Montreal, PQ / Chesterville, ON
    ...      ...  ...                              ...
    1304       3  ...                              NaN
    1305       3  ...                              NaN
    1306       3  ...                              NaN
    1307       3  ...                              NaN
    1308       3  ...                              NaN
<BLANKLINE>
    [1309 rows x 13 columns]

Note: as a shorthand for the above, we can use:

>>> X = skrub.X(features)

Now we apply some transformations
Note that when we call ``apply()`` we can specify which columns should be transformed

>>> from skrub import selectors as s
>>> from sklearn.preprocessing import OneHotEncoder

>>> X = X.skb.apply(
...     OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
...     cols=["sex", "embarked"],
... ).skb.apply(skrub.MinHashEncoder(n_components=2), cols=~s.numeric())

>>> X
<Apply MinHashEncoder>
Preview:
――――――――
          pclass        name_0        name_1  ...  embarked_Q  embarked_S  embarked_nan
    0          1 -2.116025e+09 -1.880914e+09  ...         0.0         1.0           0.0
    1          1 -2.098759e+09 -2.091214e+09  ...         0.0         1.0           0.0
    2          1 -2.129262e+09 -2.031541e+09  ...         0.0         1.0           0.0
    3          1 -2.073325e+09 -2.074258e+09  ...         0.0         1.0           0.0
    4          1 -2.135784e+09 -2.134704e+09  ...         0.0         1.0           0.0
    ...      ...           ...           ...  ...         ...         ...           ...
    1304       3 -2.131556e+09 -1.598010e+09  ...         0.0         0.0           0.0
    1305       3 -2.116025e+09 -1.729951e+09  ...         0.0         0.0           0.0
    1306       3 -2.135784e+09 -2.134704e+09  ...         0.0         0.0           0.0
    1307       3 -1.964294e+09 -2.086339e+09  ...         0.0         0.0           0.0
    1308       3 -2.085421e+09 -1.598010e+09  ...         0.0         1.0           0.0
<BLANKLINE>
    [1309 rows x 22 columns]

For supervised estimators we pass the target as the ``y`` parameter.

>>> from sklearn.ensemble import HistGradientBoostingClassifier

>>> prediction = X.skb.apply(
...     HistGradientBoostingClassifier(
...         learning_rate=skrub.choose_float(0.001, 0.1, log=True, name="lr")
...     ),
...     y=y,
... )

>>> prediction
<Apply HistGradientBoostingClassifier>
Preview:
――――――――
          y
    0     1
    1     1
    2     0
    3     0
    4     0
    ...  ..
    1304  0
    1305  0
    1306  0
    1307  0
    1308  0
<BLANKLINE>
    [1309 rows x 1 columns]

>>> print(prediction.skb.describe_steps())
VAR 'X'
APPLY OneHotEncoder
APPLY MinHashEncoder
VAR 'y'
APPLY HistGradientBoostingClassifier

>>> print(prediction.skb.describe_param_grid())
- lr: choose_float(0.001, 0.1, log=True, name='lr')

Now we can get our hyperparameter-tuning estimator based on the choices we
inserted in our expression, fit it and see the results

>>> search = prediction.skb.get_randomized_search(n_iter=2, cv=2)

>>> search.fit({"X": features, "y": target})
ParamSearch(expr=<Apply HistGradientBoostingClassifier>,
            search=RandomizedSearchCV(cv=2, estimator=None, n_iter=2,
                                      param_distributions=None))
>>> print(search.get_cv_results_table())
   mean_test_score        lr
0         0.922814  0.084082
1         0.922028  0.004061
