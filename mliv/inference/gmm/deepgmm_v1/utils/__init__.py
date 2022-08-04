from typing import Dict, Any, Iterator, Tuple
from itertools import product


def grid_search_dict(org_params: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Iterate list in dict to do grid search.

    Examples
    --------
    >>> test_dict = dict(a=[1,2], b = [1,2,3], c = 4)
    >>> list(grid_search_dict(test_dict))
    [('a:1-b:1', {'c': 4, 'a': 1, 'b': 1}),
    ('a:1-b:2', {'c': 4, 'a': 1, 'b': 2}),
    ('a:1-b:3', {'c': 4, 'a': 1, 'b': 3}),
    ('a:2-b:1', {'c': 4, 'a': 2, 'b': 1}),
    ('a:2-b:2', {'c': 4, 'a': 2, 'b': 2}),
    ('a:2-b:3', {'c': 4, 'a': 2, 'b': 3})]
    >>> test_dict = dict(a=1, b = 2, c = 3)
    >>> list(grid_search_dict(test_dict))
    [('one', {'a': 1, 'b': 2, 'c': 3})]

    Parameters
    ----------
    org_params : Dict
        Dictionary to be grid searched

    Yields
    ------
    name : str
        Name that describes the parameter of the grid
    param: Dict[str, Any]
        Dictionary that contains the parameter at grid

    """
    search_keys = []
    non_search_keys = []
    for key in org_params.keys():
        if isinstance(org_params[key], list):
            search_keys.append(key)
        else:
            non_search_keys.append(key)
    if len(search_keys) == 0:
        yield "one", org_params
    else:
        param_generator = product(*[org_params[key] for key in search_keys])
        for one_param_set in param_generator:
            one_dict = {k: org_params[k] for k in non_search_keys}
            tmp = dict(list(zip(search_keys, one_param_set)))
            one_dict.update(tmp)
            one_name = "-".join([k + ":" + str(tmp[k]) for k in search_keys])
            yield one_name, one_dict
