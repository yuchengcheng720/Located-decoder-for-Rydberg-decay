# Located-decoder-for-Rydberg-decay
code for my work about located error and Rydberg decay. Related works are available at https://arxiv.org/abs/2503.01649 and https://arxiv.org/abs/2411.04664

Code to get the results in my work about SWAP-LRU & Rydberg decay (arxiv_2503.01649) is available. The code is based on *Python*. It is mainly based on *Numpy* to generate the error, *Stim* to handle the detector error model and *Pymatching* to decode with MWPM algorithm.

*SWAP.py* file has the code and *SWAP_Example.ipynb* has some simple examples about how to use the code, including how to derive the threshold and error distance. The code now is not optimized considering the speed, the main time overhead now is *reweighting* process instead of the decoding algorithm itself.

# If you use this code, please cite the article below
```
@misc{yu2025locatingrydbergdecayerror,
      title={Locating Rydberg Decay Error in SWAP-LRU}, 
      author={Cheng-Cheng Yu and Yu-Hao Deng and Ming-Cheng Chen and Chao-Yang Lu and Jian-Wei Pan},
      year={2025},
      eprint={2503.01649},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2503.01649}, 
}
```
