import numpy as np

class Oneshot(object) :
    def __init__(self, feature_dim=5, stim_ratio=[16,8,1], stim_vectors=[0.5, 0.25, 0.1],
                 cue_ratio=[4,1], cue_vectors=[0.6, 0.3], lr=0.1, primacy=0.36, recency=0.36) :

        self.tot_stim=25
        self.stim_ratio = stim_ratio
        self.stim_vectors = stim_vectors

        self.tot_cue=5
        self.cue_ratio = cue_ratio
        self.cue_vectors = cue_vectors

        from Oneshot_sangwan import os_sangwan
        self.model = os_sangwan(lr=lr, primacy=primacy, recency = recency);

        self.tot_dim=feature_dim;


    def _spawn_stims(self,) :

        candidates = np.arange(self.tot_stim);
        np.random.shuffle(candidates)

        stims = np.zeros(shape=(self.tot_stim, self.tot_dim), dtype=np.float32);

        for sr, sr_value in zip(self.stim_ratio, self.stim_vectors) :
            idxes = np.random.choice(candidates, size = sr, replace=False);
            stims[idxes] = sr_value;

            for id in idxes :
                cand_id = np.where(candidates == id);
                candidates = np.delete(candidates, cand_id)

        stims = np.reshape(stims, (5,5, self.tot_dim));

        return stims

    def _spawn_cues(self, idx=None):

        cues = np.zeros(shape=(self.tot_cue, self.tot_dim), dtype=np.float32);

        for c_idx, c in enumerate(cues) :

            if c_idx == idx :
                cues[c_idx] = self.cue_vectors[-1];
            else :
                cues[c_idx] = self.cue_vectors[0]

        return cues

    def gen_episode(self, target=None):

        if target is None :
            target = np.random.choice([0,1], size = 1, replace=False)

        stims = self._spawn_stims();

        novel_stim_r = np.where(np.argmax(stims, -1) == 2)[0][0];

        if target == 1 :
            novel_rew_row = novel_stim_r
        else :
            candidates = np.arange(5);
            candidates = np.delete(candidates, novel_stim_r)
            novel_rew_row = np.random.choice(candidates, size=1, replace=False)

        cues = self._spawn_cues(novel_rew_row)
        ep = np.concatenate([stims, np.expand_dims(cues, 1)], axis=1);

        labels = self.model.perform(ep)

        return ep, labels, (None, None)