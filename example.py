from __future__ import division
import numpy as np
import pandas as pd
np.seterr(divide='ignore')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import pyhsmm
import models


T = 400

#FRIDGE
true_obsdistns_chain1 = [
        pyhsmm.basic.distributions.ScalarGaussianFixedvar(
            mu_0=0,tausq_0=1,sigmasq=5),
        pyhsmm.basic.distributions.ScalarGaussianFixedvar(
            mu_0=115, tausq_0=10, sigmasq=10),
        pyhsmm.basic.distributions.ScalarGaussianFixedvar(
            mu_0=425,tausq_0=30,sigmasq=10),
        ]


#DISH
true_obsdistns_chain2 = [
        pyhsmm.basic.distributions.ScalarGaussianFixedvar(
            mu_0=0, tausq_0=1, sigmasq=5),
        # pyhsmm.basic.distributions.ScalarGaussianFixedvar(
        #     mu_0=525, tausq_0=25, sigmasq=10),
        pyhsmm.basic.distributions.ScalarGaussianFixedvar(
            mu_0=900, tausq_0=200, sigmasq=10),
        ]


# observation hyperparameters used during inference
obshypparamss = [
        dict(mu_0=100.,tausq_0=50.**2,sigmasq_0=10,nu_0=100.),
        dict(mu_0=225.,tausq_0=25.**2,sigmasq_0=10,nu_0=100.),
        ]

durhypparamss = [
        dict(r=10,alpha_0=100.,beta_0=600.),
        dict(r=10,alpha_0=100.,beta_0=200.),
        ]

truemodel = models.Factorial([models.FactorialComponentHSMM(
        init_state_concentration=2.,
        alpha=2.,gamma=4.,
        obs_distns=od,
        dur_distns=[pyhsmm.basic.distributions.NegativeBinomialFixedRDuration(**durhypparams) for hi in range(len(od))])
    for od,durhypparams in zip([true_obsdistns_chain1,true_obsdistns_chain2],durhypparamss)])

sumobs, allobs, allstates = truemodel.generate(T)

plt.figure(); plt.plot(sumobs); plt.title('summed data')
plt.figure(); plt.plot(truemodel.states_list[0].museqs); plt.title('true decomposition')
plt.show();