
import numpy as np

def get_expon_weight_func(
    weight_init, weight_final, weight_delay_steps=0, weight_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is weight_init when step=0 and weight_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If weight_delay_steps>0 then the learning rate will be scaled by some smooth
    function of weight_delay_mult, such that the initial learning rate is
    weight_init*weight_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>weight_delay_steps.

    :param conf: config subtree 'weight' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (weight_init == 0.0 and weight_final == 0.0):
            # Disable this parameter
            return 0.0
        if weight_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = weight_delay_mult + (1 - weight_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / weight_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(weight_init) * (1 - t) + np.log(weight_final) * t)
        return delay_rate * log_lerp

    return helper