import numpy as np
from Oneshot import Oneshot

if __name__ == '__main__':


    feature_dim = 5;
    stim_ratio = [16, 8, 1];
    stim_vectors = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                    np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
                    np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])]
    cue_ratio = [4, 1]
    cue_vectors = [np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
                   np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]

    os_exp = Oneshot(feature_dim=feature_dim, stim_ratio=stim_ratio, stim_vectors=stim_vectors,
                     cue_ratio=cue_ratio, cue_vectors=cue_vectors, primacy=0.36, recency=0.36)

    incre_stim, incre_lr, _ = os_exp.gen_episode(target=0) # incremental case
    oneshot_stim, oneshot_lr, _ = os_exp.gen_episode(target=1)  # one-shot case
