# toolkitPython
Python port of [BYU CS 478 machine learning toolkit](http://axon.cs.byu.edu/~martinez/classes/478/stuff/Toolkit.html)

Works with Python 2.7 or 3. Requires [NumPy](http://www.numpy.org).

## Usage

In order to use this toolkit, most commands will be similar to those given
on the class website for the Java and C++ toolkits. With the assumption that
you already have NumPy installed (see their [website](http://www.numpy.org) for
installation instructions), usage is straight-forward.

As example, execute the following commands from the root directory of this
repository.

```bash
mkdir datasets
wget http://axon.cs.byu.edu/~martinez/classes/478/stuff/iris.arff -P datasets/
python -m toolkit.manager -L baseline -A datasets/iris.arff -E training
```

Aside from needing to specify the module to run, commands follow the same
syntax as the other toolkits.

For information on the expected syntax, run

```bash
python -m toolkit.manager --help
```

## Creating Learners

See the baseline_learner.py and its `BaselineLearner` class for an example of
the format of the learner. In particular, new learners will need to override
the `train()` and `predict()` functions of the `SupervisedLearner` base class.

