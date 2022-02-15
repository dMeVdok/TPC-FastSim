import numpy as np

from metrics.gaussian_metrics import get_val_metric_v as gaussian_fit


class Identity:
    def scale(self, x):
        return x
    
    def unscale(self, x):
        return x


class Logarithmic:
    def scale(self, x):
        return np.log10(1 + x)

    def unscale(self, x):
        return 10 ** x - 1


class Gaussian:
    def __init__(self, shape=(8, 16)):
        self.shape = shape

    def scale(self, x):
        result = gaussian_fit(x)
        result[:,-1] = np.log(result[:,-1]) / np.log(10)
        # result[:,3] /= (result[:,2] * result[:,4])
        return np.stack([result[:,1], result[:,0], result[:,4], result[:,3], result[:,2], result[:,5]], axis=1)

    def unscale(self, x, use_activations=False):

        if use_activations:
            sx, sy = self.shape[1]-1, self.shape[0]-1
            mu, ms, it = (
                x[:,:2],
                x[:,2:-1],
                x[:,-1],
            )
            ms = np.stack([0.001 + 0.999*(1. + np.tanh(ms[:,0])) / 2, np.tanh(ms[:,1]), 0.001 + 0.999*(1. + np.tanh(ms[:,2])) / 2], axis=1)
            ms = np.stack([(ms[:,0])*3*sy, ms[:,1]*np.sqrt((ms[:,0])*ms[:,2]*9*sx*sy), ms[:,2]*3*sx], axis=1)
            mu = np.tanh(mu) * np.array([[sx/2, sy/2]]) + np.array([[sx/2, sy/2]])
            it = np.maximum(0, it)
            x = np.stack(
                [
                    mu[:, 0],
                    mu[:, 1],
                    ms[:, 0],
                    ms[:, 1],
                    ms[:, 2],
                    it
                ], 
            axis=1)

        m1, m0, D11, D01, D00, logA = x.T
        D00 = np.clip(D00, 0.05, None)
        D11 = np.clip(D11, 0.05, None)
        
        #assert not np.any(np.abs(D01) >= 1), "f"
        #D01 *= D00 * D11

        A = 10**logA

        cov = np.stack([
            np.stack([D00, D01], axis=1),
            np.stack([D01, D11], axis=1)
        ], axis=2) # N x 2 x 2

        invcov = np.linalg.inv(cov)
        mu = np.stack([m0, m1], axis=1)

        xx0 = np.arange(self.shape[0])
        xx1 = np.arange(self.shape[1])
        xx0, xx1 = np.meshgrid(xx0, xx1, indexing='ij')
        xx = np.stack([xx0, xx1], axis=2)
        residuals = xx[None,...] - mu[:,None,None,:] # N x H x W x 2

        result = np.exp(-0.5 *
            np.einsum('ijkl,ilm,ijkm->ijk', residuals, invcov, residuals)
        )

        result /= np.sqrt(np.linalg.det(cov).reshape(cov.shape[0],1,1)*((2*np.pi)**2))
        #result /= result.sum(axis=(1, 2), keepdims=True)
        result *= A[:,None,None]

        #print(result)

        return result


def get_scaler(scaler_type):
    if scaler_type == 'identity':
        return Identity()
    elif scaler_type == 'logarithmic':
        return Logarithmic()
    elif scaler_type == 'gaussian':
        return Gaussian()
    else:
        raise NotImplementedError(scaler_type)
