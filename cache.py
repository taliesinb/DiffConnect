import opcode
import dis
import weakref
import types
import dill
import hashlib
import os.path
import time
import functools
import glob
from collections import OrderedDict


'''
This file contains a utility that allows the results of expensive functions be cached.
This is achieved by writing the results to disk, using a unique path that identifies the
particular version of the function that is being cached (based on a hash that depends on
its definition, thanks to dill), as well as a key that depends on the hashed arguments 
of the function.

Such cached functions should not contain (possibly mutually) recursive definitions, or 
the stack will overflow. Their hash values WILL change if functions that depend on are 
changed, but these  dependencies are only followed if they remain within the current module. 

The pickled results will be cached under cache/funcname/hexhash/arg_hash.dill

Each pickled result contains a dictionary with keys:
'input': ordered dict of function arguments + their provided (or defaulting) values  
'output': whatever the function produced
'time': time in seconds the function took to run
'''


# this will store hashes and global lists for functions / values,
# without preventing them from being GC'd
code_object_global_names_cache = weakref.WeakKeyDictionary()
stable_hash_cache = weakref.WeakKeyDictionary()


STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)


# adapted from https://web.archive.org/web/20140626004012/http://www.picloud.com
def get_code_object_global_names(code_object):
    global code_object_global_names_cache
    out_names = code_object_global_names_cache.get(code_object)
    if out_names:
        return out_names
    try:
        names = code_object.co_names
    except AttributeError:
        out_names = set()
    else:
        out_names = set()
        for instr in dis.get_instructions(code_object):
            op = instr.opcode
            if op in GLOBAL_OPS:
                out_names.add(names[instr.arg])
        # see if nested function have any global refs
        if code_object.co_consts:
            for const in code_object.co_consts:
                if type(const) is types.CodeType:
                    out_names |= get_code_object_global_names(const)
    out_names = sorted(out_names)
    code_object_global_names_cache[code_object] = out_names
    return out_names


# this generates the digests of all the globals that a function
# depends on
def hash_code_object_dependencies(fn, hasher, module_scope):
    code_object = fn.__code__
    global_dict = fn.__globals__
    if module_scope and fn.__module__ != module_scope:
        # print (f"skipping dependencies out-of-scope function {fn.__name__}")
        return
    for var in get_code_object_global_names(code_object):
        if var in sorted(global_dict.keys()):
            val = global_dict[var]
            digest = hash_digest(val, module_scope)
            # print(f'{fn.__name__} depends on {var} which has digest {digest}')
            hasher.update(digest)


# we don't want our hashing of functions to depend on unstable properties of the
# co, like the filename, first line, source map, etc.
def get_stable_code_object_fields(co):
    return (co.co_argcount,
            co.co_nlocals,
            co.co_flags,
            co.co_stacksize,
            co.co_names,
            co.co_varnames,
            co.co_code,
            co.co_consts)


# this hashes a value, using dill. if that value is a function, we special case
# the hashing to 1) be stable 2) include the hashes of those functions it depends on
def hash_digest(x, module_scope=None, hasher=None):
    global stable_hash_cache
    if x in stable_hash_cache:
        return stable_hash_cache.get(x)
    is_func = hasattr(x, '__code__')
    if hasher is None:
        hasher = hashlib.new('md5')
    if is_func:
        dump = dill.dumps(get_stable_code_object_fields(x.__code__))
    else:
        dump = dill.dumps(x)
    hasher.update(dump)
    # print(f"base hash for {x}: {hasher.hexdigest()}")
    if is_func:
        hash_code_object_dependencies(x, hasher, module_scope or x.__module__)
    digest = hasher.digest()
    try:
        stable_hash_cache[x] = digest
    except TypeError:
        pass
    return digest


def hash_hexdigest(x, module_scope=None):
    hasher = hashlib.new('md5')
    hash_digest(x, module_scope=module_scope, hasher=hasher)
    return hasher.hexdigest()


def load_cached_results(fn):
    cache_path = fn.__cache_path__
    for file_path in glob.glob(cache_path + '*'):
        res = unpickle(file_path)
        yield res


def load_cached_results_as_pandas(fn, exclude=None, index=None):
    import pandas
    cache_path = fn.__cache_path__
    records = []
    for file_path in glob.glob(cache_path + '*'):
        res = unpickle(file_path)
        inputs = res['input']
        outputs = res['output']
        if not isinstance(outputs, dict):
            outputs = {'output': outputs}
        record = inputs
        record.update(outputs)
        record['timing'] = res['timing']
        records.append(record)
    return pandas.DataFrame.from_records(records, exclude=exclude, index=index)



# this simulates the way that python will resolve positional and named arguments, yielding
# a single ordered dict that contains the names of arguments and their values
def normalize_args(fn_name, args, kwargs, arg_names, defaults):
    result = OrderedDict()
    for key in kwargs.keys():
        if key not in arg_names:
            raise RuntimeError(f"{fn_name}: unknown key {key} provided")
    if len(args) > len(arg_names):
        raise RuntimeError(f"{fn_name}: excess arguments provided {len(args)} > {len(arg_names)}")
    for i, name in enumerate(arg_names):
        if i < len(args):
            result[name] = args[i]
        elif name in kwargs:
            result[name] = kwargs[name]
        elif name in defaults:
            result[name] = defaults[name]
        else:
            raise RuntimeError(f"{fn_name}: value for argument {i} ('{name}') not specified")
    return result


# this is the decorator that turns a function into a disk-memoizing version
def cached(fn):

    fn.__cache_path__ = 'cache/' + fn.__name__ + '/' + hash_hexdigest(fn) + '/'
    fn.__arg_names__ = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    defaults = fn.__defaults__ or []
    fn.__kwdefaults__ = dict(zip(fn.__arg_names__[-len(defaults):], defaults))

    if not os.path.exists(fn.__cache_path__):
        os.makedirs(fn.__cache_path__)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        input_dict = normalize_args(fn.__name__, args, kwargs, fn.__arg_names__, fn.__kwdefaults__)
        # print(f"Input: {input_dict}")
        input_dump = dill.dumps(input_dict)
        hasher = hashlib.new('md5')
        hasher.update(input_dump)
        input_digest = hasher.hexdigest()
        output_path = fn.__cache_path__ + input_digest
        if os.path.exists(output_path):
            res = unpickle(output_path)
            return res['output']
        start = time.time()
        output = fn(*args, **kwargs)
        end = time.time()
        output_dict = {'input': input_dict, 'output': output, 'timing': end - start}
        # print(f"Output: {output_dict}")
        pickle(output_path, output_dict)
        return output

    return wrapper


def unpickle(path):
    with open(path, 'rb') as file:
        return dill.load(file)


def pickle(path, value):
    with open(path, 'wb') as file:
        dill.dump(value, file)


if __name__ == '__main__':

    from time import sleep

    @cached
    def double(x):
        sleep(0.1)
        return x * 2

    # this will be slow
    for i in range(10):
        double(i)

    # this will be fast
    for i in range(10):
        double(i)

    # this returns all the values
    print(list(load_cached_results(double)))