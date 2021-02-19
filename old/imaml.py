import torch, torch.nn.functional
import indent
import hyper
import cg
import utils

run_count = 0

def imaml_Ax(𝓛, θ, λ=1.0, regularization=1.0):
    return lambda x: (1 / λ) * utils.hessian_vector_product(𝓛, θ, x) + (1.0 + regularization) * x

@indent.indenting
def train(slow_net, iterator_factory, *, 
        loss_fn=utils.cross_entropy_loss,
        slow_method='Adam', slow_lr=0.1, 
        fast_method='SGD', fast_lr=0.1, 
        log_dir='runs', title=None,
        batch_size=256, flatten=True,
        fast_steps=5, slow_steps=20, num_tasks=10):
    writer = utils.make_log_writer(log_dir, title)
    training_iterator, test_iterator = iterator_factory(batch_size=batch_size, flatten=flatten)
    loss_history = []
    acc_history = []
    if not hasattr(slow_net, 'target_net'):
        slow_net = hyper.DummyHyperNetwork(slow_net)
    slow_params = slow_net.parameters()
    slow_optimizer = utils.opt_mapping[slow_method](slow_params, lr=slow_lr)
    fast_net = slow_net.target_net
    fast_params = slow_net.target_params
    fast_optimizer = utils.opt_mapping[fast_method](fast_params, lr=fast_lr)
    meta_grad_vector = torch.zeros(sum(a.numel() for a in fast_params)) # create a single vector to store metagrads of all fast params
    meta_grads = utils.split_as(meta_grad_vector, fast_params) # expose this storage via a list of arrays (as views)
    # helper function that trains the fast network from scratch on a batch from the given generator
    print('hello')
    def fast_train(generator):
        slow_net.push_weights() # copy the most recent weights onto the fast learner
        for fast_step in range(fast_steps): # fast learning loop
            loss = loss_fn(fast_net, next(training_iterator))# + 0.05 * slow_net.discrepancy_loss(target_grad=True)
            #print(loss.item(), ' ',)
            fast_net.zero_grad()
            loss.backward()
            fast_optimizer.step()
        loss = 0.1 * slow_net.discrepancy_loss(target_grad=True)
        loss.backward()
        fast_optimizer.step()

    for step in range(slow_steps): # slow learning loop
        meta_grad_vector.zero_() # zero the meta gradients
        slow_net.forward() # derive θ_init
        total_final_loss = total_final_accuracy = 0
        do_accuracy = (step % 5 == 0)
        for task_id in range(num_tasks): # task loop
            fast_train(training_iterator) # train the fast net from θ_init to obtain θ_final
            final_loss = loss_fn(fast_net, next(test_iterator))  # calculate (fresh) non-regularized loss at θ_final
            final_grads = torch.autograd.grad(final_loss, fast_params, retain_graph=True) # θ-gradient 
            total_final_loss += final_loss.item()
            meta_grad_vector += cg.solve(\
                imaml_Ax(final_loss, fast_params), # lambda that produces H.v products for CG solution of Ax = b\
                utils.to_vector(final_grads) # value of b \
            )
            if do_accuracy:
                total_final_accuracy += utils.test_accuracy(fast_net, test_iterator) 
        slow_net.zero_grad()
        slow_net.backward(grads=meta_grads) # propogate the fast metagradients backward through the slow net    
        for k in range(100):
            slow_optimizer.step() # do one step of slow optimization
        final_loss = total_final_loss / num_tasks
        if writer: writer.add_scalar("final_loss", final_loss, step)
        if do_accuracy:
            # utils.print_model_parameters(slow_net)
            final_accuracy = total_final_accuracy / num_tasks
            print(f"{step:>5d}\t{final_loss:.3f}\t{final_accuracy:.3f}")
            if writer: writer.add_scalar("final_accuracy", final_accuracy, step)
        else:
            print(f"{step:>5d}\t{final_loss:.3f}")
        
    if writer: writer.close()

