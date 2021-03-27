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

from types import ModuleType, CodeType
from pprint import pprint
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

# objects that live in these modules will have a hash that is simply their full path,
# rather than their transitive hashes
shallow_hash_modules = [
    'colorama', 'indent', 'data', 'cache',
    'numpy', 'torch', 'pandas',
    'tensorboardX', 'h5py', 'dill',
    'random', 'math', 'itertools', 'functools', 'sys', 'io', 'builtins',
    'operator', 'datetime', 'time', 'pathlib', 'os', 'filecmp'
]

NAME_OPS = tuple(opcode.opmap[name] for name in ['STORE_GLOBAL', 'DELETE_GLOBAL', 'LOAD_GLOBAL', 'LOAD_METHOD', 'LOAD_ATTR'])
cache_logging = False

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
            if op in NAME_OPS:
                try:
                    const = names[instr.arg]
                    out_names.add(const)
                except:
                    pass
        # see if nested function have any global refs
        if code_object.co_consts:
            for const in code_object.co_consts:
                if type(const) is types.CodeType:
                    out_names |= set(get_code_object_global_names(const))
    out_names = sorted(out_names)
    code_object_global_names_cache[code_object] = out_names
    return out_names

def hash_module_functions(mod, global_names, hasher):
    count = 0
    mod_name = mod.__name__
    for name in global_names:
        if hasattr(mod, name):
            elem = getattr(mod, name)
            if getattr(elem, '__module__', None) == mod_name and hasattr(elem, '__code__'):
                count += 1
                if cache_logging:
                    print(f"found possible match {name} in module {mod.__name__}")
                # this is a shallow hash of just the function's code, and not its dependancies
                hasher.update(dill.dumps(stabilize(elem)))
    if count == 0 and cache_logging:
        print(f'ignoring module {mod_name}, which has no attrs that match global names')

# this generates the digests of all the globals that a function
# depends on
def hash_code_object_dependencies(fn, hasher, module_scope):
    code_object = fn.__code__
    global_dict = fn.__globals__
    fn_module = fn.__module__
    global_keys = sorted(global_dict.keys())
    if module_scope and fn_module != module_scope:
        if cache_logging:
            print (f"skipping dependencies out-of-scope function {fn.__name__}")
        return
    global_names = get_code_object_global_names(code_object)
    for var in global_names:
        if var in global_keys:
            val = global_dict[var]
            if isinstance(val, ModuleType) and val.__name__ != fn_module:
                hash_module_functions(val, global_names, hasher)
            else:
                digest = hash_digest(val)
                if cache_logging:
                    print(f'{fn.__name__} depends on {var} ({type(val)}) which has digest {digest}')
                hasher.update(digest)

# we don't want our hashing of functions to depend on unstable properties of the
# co, like the filename, first line, source map, etc.
def get_stable_fields(co):
    return (
        co.co_argcount,
        co.co_nlocals,
        co.co_flags & ~1,   # null out the 'optimized' flag
        co.co_names,
        co.co_varnames,
        co.co_code,
        tuple(stabilize(const) for const in co.co_consts)
    )

def stabilize(obj):
    if hasattr(obj, '__stabilized_code__'):
        return getattr(obj, '__stabilized_code__')
    if hasattr(obj, '__code__'):
        if cache_logging:
            print("stabilizing ", obj.__name__)
        stable_fields = get_stable_fields(getattr(obj, '__code__'))
        setattr(obj, '__stabilized_code__', stable_fields)
        return stable_fields
    if isinstance(obj, CodeType):
        return get_stable_fields(obj)
    return obj

def get_module_path(x):
    if hasattr(x, '__loader__') and hasattr(x, '__name__'):
        module_path = getattr(x, '__name__')
    elif hasattr(x, '__module__'):
        module_path = getattr(x, '__module__')
        if hasattr(x, '__name__'): module_path += '.' + getattr(x, '__name__')
    else:
        return None, None
    return module_path.split('.', 1)[0], module_path

# this hashes a value, using dill. if that value is a function, we special case
# the hashing to 1) be stable 2) include the hashes of those functions it depends on
# we will recurse over tuples and dicts, in case one of their elements is also a function.
def hash_digest(x, module_scope=None, hasher=None):
    global stable_hash_cache
    print("calling hash_digest on ", x)
    if x in stable_hash_cache:
        if cache_logging: print(x, ' is cached ')
        return stable_hash_cache.get(x)
    base_module_path, full_module_path = get_module_path(x)
    if base_module_path in shallow_hash_modules:
        if cache_logging: print(f"using shallow hash for object with path '{full_module_path}' of type {type(x)}")
        full_module_path = full_module_path.encode('utf-8')
        stable_hash_cache[x] = full_module_path
        return full_module_path
    if hasattr(x, '__manual_hash__'):
        manual_hash = getattr(x, '__manual_hash__')
        if cache_logging: print(f"using manual hash with value {manual_hash}")
        stable_hash_cache[x] = manual_hash
        return manual_hash
    if hasher is None:
        hasher = hashlib.new('md5')
    deps_queue = []
    def stabilize_and_queue(value):
        if hasattr(value, '__code__'):
            deps_queue.append(value)
            return stabilize(value)
        return value
    hasher.update(dill.dumps(container_recurse(stabilize_and_queue, x)))
    if cache_logging:
        print(f"base hash for {x}: {hasher.hexdigest()}")
    for q in deps_queue:
        hash_code_object_dependencies(q, hasher, module_scope or q.__module__)
    digest = hasher.digest()
    try:
        print(f'putting {x} in stable hash cache')
        stable_hash_cache[x] = digest
    except TypeError:
        print("failed to hash", x)
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

def container_recurse(f, value):
    if isinstance(value, tuple):
        return tuple(container_recurse(f, v) for v in value)
    if isinstance(value, list):
        return list(container_recurse(f, v) for v in value)
    if isinstance(value, dict):
        return dict((container_recurse(f, k), container_recurse(f, v)) for k, v in value.items())
    return f(value)

def load_cached_results_as_pandas(fn, exclude=None, index=None, namify=True):
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
        if namify:
            record = container_recurse(lambda e: getattr(e, '__name__', e), record)
        records.append(record)
    return pandas.DataFrame.from_records(records, exclude=exclude, index=index)

def load_multiple_cached_results_as_pandas(fn_list):
    datasets = []
    for fn in fn_list:
        dataset = load_cached_results_as_pandas(fn)
        dataset['label'] = fn.__name__
        datasets.append(dataset)
    return pd.concat(datasets)

# this simulates the way that python will resolve positional and named arguments, yielding
# a single ordered dict that contains the names of arguments and their values
def normalize_args(fn_name, args, kwargs, arg_names, defaults):
    result = OrderedDict()
    for key in kwargs.keys():
        if key not in arg_names:
            if key == 'global_seed':
                result['global_seed'] = kwargs[key]
            elif key in defaults:
                result[key] = kwargs[key]
            else:
                raise RuntimeError(f"{fn_name}: unknown key {key} provided. Must be one of {arg_names} or {defaults}")
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


def apply_global_seed(seed):
    import numpy
    numpy.random.seed(seed)
    import torch
    torch.manual_seed

# this is the decorator that turns a function into a disk-memoizing version
def cached(fn, manual_hash=None):
    global cache_logging

    if isinstance(manual_hash, str):
        fn_hash = manual_hash
    elif manual_hash is None:
        fn_hash = hash_hexdigest(fn)
    else:
        cache_logging = True
        print('current function hash: ', hash_hexdigest(fn))
        raise Exception("manual hash should be a str or None")

    __cache_path__ = 'cache/' + fn.__name__ + '/' + fn_hash + '/'
    __arg_names__ = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    defaults = fn.__defaults__ or []
    kwdefaults = fn.__kwdefaults__ or dict(zip(__arg_names__[-len(defaults):], defaults))

    if not os.path.exists(__cache_path__):
        os.makedirs(__cache_path__)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        input_dict = normalize_args(fn.__name__, args, kwargs, __arg_names__, kwdefaults)
        input_dict = container_recurse(stabilize, input_dict)
        if cache_logging:
            print(f"Input: {input_dict}")
        input_dump = dill.dumps(input_dict)
        hasher = hashlib.new('md5')
        hasher.update(input_dump)
        input_digest = hasher.hexdigest()
        output_path = __cache_path__ + input_digest
        if os.path.exists(output_path):
            res = unpickle(output_path)
            return res['output']
        start = time.time()
        #
        if 'global_seed' in kwargs:
            apply_global_seed(kwargs['global_seed'])
            if 'global_seed' not in __arg_names__:
                del kwargs['global_seed']
        output = fn(*args, **kwargs)
        end = time.time()
        output_dict = {'input': input_dict, 'output': output, 'timing': end - start}
        if cache_logging:
            print(f"Output: {output_dict}")
        pickle(output_path, output_dict)
        return output

    wrapper.cached_results = lambda: load_cached_results(fn)

    return wrapper


def unpickle(path):
    with open(path, 'rb') as file:
        return dill.load(file)


def pickle(path, value):
    with open(path, 'wb') as file:
        dill.dump(value, file)


if __name__ == '__main__':

    from time import sleep

    cache_logging = True

    def zint(x):
        print("Z", x)

    print("hash of func:")
    print(hash_digest(zint))

    print("hash of func in tuple:")
    print(hash_digest((zint, 3, 4)))

    @cached
    def double(x):
        sleep(0.5)
        print('foo')
        return x * 2

    print("uncached (will be slow)")
    for i in range(5):
        double(i)

    print("cached (will be fast)")
    for i in range(5):
        double(i)

    print("uncached (will be slow, unique global seed)")
    for seed in range(5):
        double(0, global_seed=seed)

    print("cached (will be fast, reuse global seed)")
    for seed in range(5):
        double(0, global_seed=seed)

    print("all cached values")
    pprint(list(load_cached_results(double)))