# reading project: survey on accelerated numpy

## source

- [legate numpy paper](https://legion.stanford.edu/pdfs/legate-preprint.pdf)
- [cupy paper](http://learningsys.org/nips17/assets/papers/paper_16.pdf)
- [dask documentation](https://docs.dask.org/en/latest/why.html)

## issues

- [stackoverflow: cupy slicing is much slower than numpy slicing](https://stackoverflow.com/questions/60956806/slicing-a-300mb-cupy-array-is-5x-slower-than-numpy) potentially due to bad memory access pattern on GPU.
