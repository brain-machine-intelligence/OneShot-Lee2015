import numpy as np

class os_sangwan(object) :
    def __init__(self, lr=0.1, primacy=0.36, recency=0.36):
        self.lambdas = [0.3333, 0.3333, 0.3333]
        self.temperature = 255.;

        self.stim_type = 3
        self.param_lr = lr
        self.param_primacy = primacy
        self.param_recency = recency

    def perform(self, exps) :
        exps = exps.argmax(-1);

        tot_alphas = [np.asarray([1.0, 1.0, 1.0])]
        tot_oslrs = []
        coupling = [-1,-1,-1];

        for r_idx, exp_row in enumerate(exps) :
            saliency = self.saliency_estimation(exps[r_idx])

            ## update coupling
            for s in range(self.stim_type) :
                if saliency[s] > 0.0 and coupling[s] == -1 :
                    coupling[s] = exp_row[-1]

            if r_idx == 0 :
                alphas = [1.0, 1.0, 1.0]
                tot_alphas.append(np.stack(alphas));

            os_lr = self.lr_estimation(alphas)
            tot_oslrs.append(np.stack(os_lr))

            rew_val = 10.0 if exps[r_idx][-1] == 4 else -10.0
            rew_val/=10

            # calculate d_alpha
            for s in range(self.stim_type) :
                primacy_effect = 1.0 if exp_row[0] == s else 0.0;
                recency_effect = 1.0 if exp_row[-2] == s else 0.0;


                d_alpha = os_lr[s] * rew_val * (1.0*saliency[s] +
                                      self.param_primacy*primacy_effect +
                                      self.param_recency*recency_effect)
                alphas[s] += self.param_lr * d_alpha

                alphas[s] = max(alphas[s],0);

            tot_alphas.append(np.stack(alphas));

        os_lr = self.lr_estimation(alphas) ## put last lr
        tot_oslrs.append(np.stack(os_lr))

        return np.stack(tot_oslrs)

    def lr_estimation(self, alphas):

        alphas = [max(alpha,0) for alpha in alphas]
        alpha_z = sum(alphas)+1e-8;

        cus = []
        for alpha in alphas :
            cu = (alpha*(alpha_z - alpha))/((alpha_z**2)*(alpha_z + 1))
            cus.append(cu)

        sm_denorm = sum([np.exp(self.temperature*c) for c in cus]);

        lr = [np.exp(self.temperature*c)/sm_denorm for c in cus]

        return lr


    def saliency_estimation(self, exp_row):

        stims = exp_row[:5];

        saliency = []

        for s_i in range(3) :
            idxes = np.where(stims == s_i)[0]
            saliency.append(min(len(idxes),1))

        return saliency
