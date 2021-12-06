# reading project: survey on accelerated numpy

## source

- legate [DOI](https://dl.acm.org/doi/10.1145/3295500.3356175) [pdf](https://legion.stanford.edu/pdfs/legate-preprint.pdf)
- cupy [pdf](http://learningsys.org/nips17/assets/papers/paper_16.pdf)
- dask [link](https://conference.scipy.org/proceedings/scipy2015/matthew_rocklin.html) [pdf](https://conference.scipy.org/proceedings/scipy2015/pdfs/matthew_rocklin.pdf)
- theano [pdf](http://conference.scipy.org/proceedings/scipy2010/pdfs/bergstra.pdf)
- numba [DOI](https://doi.org/10.1145/2833157.2833162)

## issues

- [stackoverflow: cupy slicing is much slower than numpy slicing](https://stackoverflow.com/questions/60956806/slicing-a-300mb-cupy-array-is-5x-slower-than-numpy) potentially due to bad memory access pattern on GPU.
