# hyperparam_opt
Code for optimizing hyperparameters of Scikit-learn models

* Less restrictive than scikit's [GridSearchCV/RandomSearchCV](http://scikit-learn.org/stable/modules/grid_search.html)
* Outputs CSV with each iteration's hyperparameter choices as well as metrics. Easy to add your own metrics

The abstraction is leaky but the code is short and you should be able to modify it for your needs pretty easily
