# deep learning coursework

For reading course with Dr. David Balduzzi at VUW.

### Interesting things:
- [Glorot pictures](https://github.com/PFCM/odds-and-ends/blob/master/Glorot%20pictures.ipynb)
  - re-producing and extending some figures from
    [this classic paper](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) with regard to initialisation
    of deep nets.
  - contains histograms of activations of various
    layers in a 5-layer network with different
    initialisation schemes and different
    non-linearities.
- [Saxe figures](https://github.com/PFCM/odds-and-ends/blob/master/saxe_figure.ipynb)
  - re-creates and extends a key figure in
    [this paper](https://arxiv.org/abs/1312.6120) on learning in linear
    nets.
  - final figure shows how fast various modes of the network's
    input/output correlation matrix start to line up with the
    correlation matrix of the data for a number of initialisations
    and activation functions.
  - the conclusions are:
    - orthogonal initialisation is good
    - linear nets learn fast (but obviously can't do as well in the
      long run).
- [Linear regions](https://github.com/PFCM/odds-and-ends/blob/master/linear_regions.ipynb)
  - [these](https://arxiv.org/abs/1402.1869)
    [papers](https://arxiv.org/abs/1312.6098)
    suggest deep ReLU networks are better than shallow ReLU networks
    because they are able to express more linear regions in their
    output with a similar number of parameters.
  - this is related to some
    [more recent work](https://arxiv.org/abs/1509.08101)
    discussing expressivity as a function of depth.
  - this notebook simply attempts to count the linear regions, fairly
    roughly, to see if there is a clear difference between shallow and
    deep nets with similar number of parameters.
  - results were checked at initialisation but also during training
    to approximate a smooth function (which requires many linear
    regions to accurately approximate).
  - results suggest there is a "sweet spot" for a given number of
    parameters -- if the net is deep but very thin the ReLUs seem
    to cancel each other out.
- [Switching Tensor Train](https://github.com/PFCM/odds-and-ends/blob/master/switching%20tensor%20train.ipynb)
  - builds on [Tensor Switching     
    Networks](https://arxiv.org/pdf/1610.10087v1.pdf)
    but utilises the popular tensor-train decomposition to reduce
    the size of the network.
  - also attempts some novel methods of adjusting the network to allow
    it to be trained by standard gradient descent.
  - results in quite a novel architecture, although experiments are
    a bit limited by time/resources.
