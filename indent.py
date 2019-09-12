import functools
import sys
from io import StringIO
import builtins


# this is the decorator that ensures all prints within the function indent by one
# extra level
def indenting(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with indent:
            return fn(*args, **kwargs)

    return wrapper


_counter = 0        # ensure neighboring prints with same indent show divider
_level = 0          # start with no indent


# this class is the context manager for ensuring prints are indented
class indenter():
    def __enter__(self):
        global _level, _counter
        _level += 1
        _counter += 1
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _level, _counter
        _level -= 1
        _counter += 1


indent = indenter()

_no_end = False         # to make sure we don't indent continued lines (prints following print(..., end='')
_last_info = (0, 0)     # to check whether we have consecutive lines from two separate scopes at same indentation


def iprint(*args, indent=0, **kwargs):
    global _no_end, _last_info
    skip_tab, _no_end = _no_end, kwargs.get('end') == ''
    if kwargs.get('file') or _level == 0 or skip_tab:
        if _level == 0: _last_info = (_counter, _level)
        return _print(*args, **kwargs)
    io = StringIO()
    _print(*args, file=io, **kwargs)
    tabs = '\t' * (_level + indent)
    info, _last_info = _last_info, (_counter, _level)
    if info[0] != _last_info[0] and info[1] == _last_info[1]:
        sys.stdout.write(tabs + '----\n')
    out_str = ''.join(map(lambda l: tabs + l, io.getvalue().splitlines(True)))
    sys.stdout.write(out_str)


_print = builtins.print      # save old print function
builtins.print = iprint      # override the default 'print' function


if __name__ == '__main__':

    @indenting
    def recur(i):
        print(f'begin({i})')
        if i > 0: recur(i - 1)
        if i > 0: recur(i - 1)
        print(f'end({i})')

    with indent:
        print('abc')
        print('def')

    with indent:
        print('abc', end='')
        print('def')

    with indent:
        print('abc')

    print('def')
    with indent:
        print('abc')
        print('def')
        recur(2)

