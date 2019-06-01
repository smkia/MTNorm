from core.gp.gp_base import GP


class GPBASE(GP):

    def LML(self,hyperparams):
        """
        calculate the log marginal likelihood for the given logtheta

        Input:
        hyperparams: dictionary
        """
        LML = self._LML_covar(hyperparams)

        if self.prior!=None:
            LML += self.prior.LML(hyperparams)
        return LML
