from functools import partial
import numpy as np
import logging as log
import random


# This is extension of the pyswarm. Need to fork the project and upadte in GIT
# https://github.com/tisimst/pyswarm
# Improved the logging feature and added implementation of SPSO


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def var_pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False, segments = 2):

    total_iter_segmented = 0
    log.basicConfig(level=log.INFO)
    #log.basicConfig(level=log.DEBUG)

    log.info("Solving the function using standard PSO")
    non_segmented_results = pso(func, lb, ub, ieqcons, f_ieqcons, args, kwargs,
                             swarmsize, omega, phip, phig, maxiter*segments,
                             minstep, minfunc, debug, processes,
                             particle_output)
    # Segmenting the problem
    log.info("PSO result for non-segmented is {:}".format(non_segmented_results))
    non_segmented_global_min = non_segmented_results[2]
    log.info("Non-segmented PSO global min {:}".format(non_segmented_global_min) )
    log.info("Solving the function using Segmented PSO")

    dim_tobe_segmented = random.randint(1,len(lb)-1)
    log.info("Dimension to be segmented : {:} ".format(dim_tobe_segmented))
    act_lb = lb[dim_tobe_segmented]
    act_ub = ub[dim_tobe_segmented]
    stepsize = (act_ub - act_lb) / segments
    #swarmsize = int(swarmsize / 2)
    #maxiter = int(maxiter/segments)
    segmented_local_mins = np.ones(segments) * np.inf
    segmented_results=[None]* segments
    for j in range(1,segments+1):
        lb[dim_tobe_segmented] = act_lb + (j -1)*stepsize
        ub[dim_tobe_segmented] = act_lb + (j)*stepsize
        #log.debug("New lower bound : " , lb )
        #log.debug("New upper bound : ", ub)
        #log.debug((func, lb, ub, ieqcons, f_ieqcons, args, kwargs,
        #          swarmsize, omega, phip, phig, maxiter,
        #          minstep, minfunc, debug, processes,
        #          particle_output))
        result = (pso(func, lb, ub, ieqcons, f_ieqcons, args, kwargs,
                  swarmsize, omega, phip, phig, maxiter,
                  minstep, minfunc, debug, processes,
                  particle_output))
        log.debug("PSO result for segment {:}".format(j) + " is {:}".format(result))
        segmented_results[j-1] = result
        segmented_local_mins[j-1] = result[2]
        total_iter_segmented = total_iter_segmented + result[-1]
    #log.INFO("Local Minimum values in each segments : ")
    log.debug(segmented_local_mins )
    global_min_index = np.argmin(segmented_local_mins)
    segmented_global_min = segmented_local_mins[global_min_index]
    log.debug( segmented_global_min)
    log.info("Segmented PSO global min {:}".format(segmented_results[global_min_index]))
    log.info("Segment number which has global min is {:} ".format( global_min_index) )


    lb[dim_tobe_segmented] = act_lb
    ub[dim_tobe_segmented] = act_ub

    segment_won = 0
    non_segment_won = 1

    if(segmented_global_min < non_segmented_global_min):
        segment_won = 1
        non_segment_won = 0

    segmented_lead = (segmented_global_min - non_segmented_global_min)
    print("Total number of iterations in PSO Segmented :" , total_iter_segmented )
    return  segmented_global_min, non_segmented_global_min , segment_won , non_segment_won, - segmented_lead


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
        particle_output=False):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    obj : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'

    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

#  TODO: Find why abs value is taken and purpose of vhigh, vlow
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
# TODO: How does the initialization takes palce?
    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)
    assert hasattr(func, '__call__'), 'Invalid function handle'

    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            log.debug('No constraints given.')
            cons = _cons_none_wrapper
        else:
            log.debug('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        log.debug('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S) * np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x * (ub - lb)

    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()

    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################

    for it in range(maxiter):
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x * (~np.logical_or(maskl, masku)) + lb * maskl + ub * masku

        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:

            if (it % 10 == 0):
                log.debug('New best for swarm at iteration {:}: {:} {:}' \
                      .format(it, fp[i_min], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min) ** 2))
            step_min = np.abs(fg - fp[i_min])
            if (it % 10 == 0):
                log.debug("Current Step size{:}".format(stepsize))
                log.debug("Current Min step{:}".format(step_min))

            if  step_min<= minfunc:

                Stopping_reason = "Stopping search: Swarm best objective change less than {:}".format(minfunc)
                log.debug(Stopping_reason)

                if particle_output:
                    return p_min, fp[i_min], p, fp, Stopping_reason, it
                else:
                    return p_min, Stopping_reason, fp[i_min], it

            elif stepsize <= minstep:

                Stopping_reason = "Stopping search: Swarm best objective change less than {:}".format(minstep)
                log.debug(Stopping_reason)

                if particle_output:
                    return p_min, fp[i_min], p, fp , Stopping_reason, it
                else:
                    return p_min,Stopping_reason, fp[i_min], it
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if(it%10 ==0 ):
            log.debug("Epoch = %.2f" % it + " best error = %.20f" % fg)
        it += 1

    Stopping_reason = 'Stopping search: maximum iterations reached --> {:}'.format(maxiter)
    log.debug(Stopping_reason)


    if not is_feasible(g):
        log.info("However, the optimization couldn't find a feasible design. Sorry")


    if particle_output:
        return g, fg, p, fp, Stopping_reason, it
    else:
        return g, Stopping_reason, fg, it