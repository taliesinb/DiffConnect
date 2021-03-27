import torch, torch.nn.functional
import numpy as np
import hyper
import utils
from indent import indent
import cache
from colorama import Fore, Back, Style
import random
import math

#@indent.indenting
def train(net, iterator_factory, *,
        max_batches=2000,
        loss_fn=utils.cross_entropy_loss,
        optimizer='Adam', lr=0.01,
        log_dir='runs', name=None,
        batch_size=64, flatten=True,
        callback_fn=None, fields=None,
        max_stagnation=1,
        min_accuracy_improvement=0.005,
        print_interval=None,
        test_interval=None,
        max_parameters=None
    ):

    if print_interval is None:
        print_interval = min(max(max_batches // 10, 250), 1000)

    if test_interval is None:
        test_interval = min(max(max_batches // 5, 1000), 4000)

    # if the net is actually a net factory, create the net from the factory
    if utils.is_factory(net):
        name = name or getattr(net[0], '__name__', None)
        net = utils.run_factory(net)

    if net is None:
        print(Fore.MAGENTA, 'model is None', Fore.RESET)
        return None

    if max_parameters is not None and utils.count_parameters(net) > max_parameters:
        print(Fore.MAGENTA, 'model paramcount', utils.count_parameters(net), 'exceeds max of', max_parameters, Fore.RESET)
        return None

    # if a hypernetwork was provided, we don't directly train via the network parameters,
    # we train via the hypernetwork parameters (which produce the network parameters)
    if isinstance(net, hyper.HyperNetwork):
        hyper_net, net = net, net.target_net
        params = hyper_net.parameters()
    else:
        hyper_net = None
        params = net.parameters()

    params = list(params)
    weight_shapes = [tuple(p.shape) for p in params]
    weight_param_count = sum([utils.product(shape) for shape in weight_shapes])
    title_str = f"Training \'{name}\'" if name else 'Training'
    shape_str = ' '.join(utils.to_shape_str(x) for x in weight_shapes)
    print(Fore.GREEN, f"{title_str}", Fore.RESET, f"(arrays: {len(params)}, params: {weight_param_count}, shapes: {shape_str})")

    writer = utils.make_log_writer(log_dir, name)
    optimizer = utils.opt_mapping[optimizer](params, lr=lr)

    training_iterator, test_iterator = iterator_factory(batch_size=batch_size, flatten=flatten)

    running_loss = None
    acc_history = []
    acc = best_acc = 0

    callback_interval = 1
    callback_results = []
    if isinstance(callback_fn, tuple) and len(callback_fn) == 2:
        callback_fn, callback_interval = callback_fn

    # number of accuracy tests since the last accuracy improvement
    stagnation = 0
    last_acc_batch = -1
    did_stagnate = False

    # main training loop
    for batch_num in range(max_batches+1):
        optimizer.zero_grad()
        if hyper_net:  # use hypernetwork (if any) to derive the parameters for our network
            net.zero_grad()
            hyper_net.zero_grad()
            hyper_net.forward()
            hyper_net.push_weights()
        loss = loss_fn(net, next(training_iterator))
        loss.backward()
        if hyper_net: # backprop from ordinary grads to hypergrads
                hyper_net.backward()
        optimizer.step()

        # report losses, etc.
        loss = loss.item()
        running_loss = (0.95 * running_loss + 0.05 * loss) if running_loss else loss

        if batch_num % 10 == 0 and writer:
            writer.add_scalar("loss", running_loss, batch_num)

        if batch_num % print_interval == 0 or batch_num % test_interval == 0:
            if batch_num % test_interval == 0:
                acc = utils.test_accuracy(net, test_iterator)
                last_acc_batch = batch_num
                best_acc = max(best_acc, acc)
                if acc == best_acc:
                    stagnation = 0
                elif acc < best_acc + min_accuracy_improvement:
                    stagnation += 1
                print(f"{batch_num:>6d}\t{running_loss:.3f}\t{acc:.3f}")
                acc_history.append((batch_num, acc))
                if writer: writer.add_scalar("accuracy", acc, batch_num)
                if stagnation > max_stagnation:
                    did_stagnate = True
                    break
            else:
                print(f"{batch_num:>6d}\t{running_loss:.3f}")

        if callback_fn and batch_num % callback_interval == 0:
            callback_info = {
                'net': net,
                'hyper_net': hyper_net,
                'batch_num': batch_num,
                'loss': loss,
                'running_loss': running_loss,
                'training_iterator': training_iterator,
                'test_iterator': test_iterator,
                'acc_history': acc_history,
            }
            callback_res = callback_fn(callback_info)
            if callback_res is not None:
                callback_results.append(callback_res)

    # report final test accuracy
    if batch_num > last_acc_batch:
        acc = utils.test_accuracy(net, test_iterator, max_items=10000)
        if writer:
            writer.add_scalar("accuracy", acc, batch_num)
            writer.close()
        print(f"{batch_num:>6d}\t{running_loss:.3f}\t{acc:.3f}")

    stagnate_message = ' (stopped early)' if did_stagnate else ''
    print(Fore.LIGHTBLUE_EX, f"Done training" + stagnate_message, Fore.RESET)

    # return a bunch of statistics about the training run
    result = {
        'final_loss': running_loss,         # final loss
        'final_accuracy': acc,              # final test accuracy
        'best_accuracy': best_acc,          # best test accuracy
        'accuracy_history': acc_history,    # history of accuracies
        'weight_shapes': weight_shapes,     # list of shapes of trained arrays
        'weight_param_count': weight_param_count,  # total number of weight parameters
        'batch_size': batch_size,           # batch size
        'batches': max_batches,
        'callback_results': callback_results,
        'name': name
    }

    if callback_fn is None:
        del result['callback_results']

    if fields is not None:
        return dict((k, result[k]) for k in fields)

    return result

cached_train = cache.cached(train, manual_hash="cached_train_0")

'''
this wraps a parameter that will be varied across
'''
class Varying:
    def __init__(self, values:list, shuffle=False, subsample=None):
       self.values = utils.shuffle_and_subsample(values, shuffle, subsample)

def filter_varying(params):
    fixed = {k: v for k, v in params.items() if not isinstance(v, Varying)}
    varying = {k: v.values for k, v in params.items() if isinstance(v, Varying)}
    return fixed, varying

TrueOrFalse = Varying([False, True])

def LogIntegerRange(a:int, b:int, num:int):
    return Varying(list(np.logspace(np.log10(a),np.log10(b), num=num, dtype='int')))

def IntegerRange(a:int, b:int, shuffle=False, subsample=None):
    if a > b:
        r = range(a, b-1, -1)
    else:
        r = range(a, b+1, 1)
    return Varying(list(r), shuffle=shuffle, subsample=subsample)

def train_models(model_list, model_settings, iterator, runs=1, train_fn=cached_train, shuffle=False, subsample=None, **kwargs):
    results = []
    if not isinstance(model_list, list):
        model_list = [model_list]
    fixed_settings, varying_settings = filter_varying(model_settings)
    settings_product = utils.shuffle_and_subsample(utils.cartesian_product(varying_settings), shuffle, subsample)
    def single_run(seed):
        for setting in settings_product:
            print(Fore.YELLOW, f"{setting}", Fore.RESET)
            for model in model_list:
                with indent:
                    res = train_fn((model, {**setting, **fixed_settings}), iterator, global_seed=seed, **kwargs)
                    if res is None:
                        continue
                    if not isinstance(res, dict):
                        res = {'result': res}
                    if 'name' not in setting and 'name' not in res and hasattr(model, '__name__'):
                        res['name'] = model.__name__
                    results.append({**setting, **res, 'global_seed':seed})
    if runs > 1:
        for seed in range(runs):
            with indent:
                print(Fore.RED, f"Run #{seed}", Fore.RESET)
                single_run(seed)
    else:
        single_run(0)
    return results

if __name__ == '__main__':

    from data import mnist

    net = utils.Linear(28*28, 10)
    res = train(net, mnist, max_batches=1000)
    print(res)