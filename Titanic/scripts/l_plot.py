#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:30:16 2018

@author: isabellekluser
"""


import seaborn as sns
import matplotlib.pyplot as plt


def titanic_survived(dataset):
    sns.set(style="darkgrid")

    
    # Make a custom palette with gendered colors
    pal = dict(male="#6495ED", female="#F08080")
    
    # Show the survival proability as a function of age and sex
    g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=dataset,
                   palette=pal, y_jitter=.02, logistic=True)
    g.set(xlim=(0, 80), ylim=(-.05, 1.05))
    
    
    g = sns.FacetGrid(dataset, row="sex", col="pclass", hue="survived", margin_titles=True)
    g.map(plt.hist, "age", alpha=0.5)
    
    plt.show()
