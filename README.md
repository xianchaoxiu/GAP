# GAP

The code in this toolbox implements "GAP: Gradient-guided Adaptive Pruning for Large Language Models" by <i>C. Huang, L. Xu, X. Xiu</i>.

### Demo
run main.py for reproduction.
```
python3 main.py --orientation gap --model meta-llama/Llama-3.2-1B --cal-dataset wikitext2 --cal-nsamples 128 --sparsity 0.20  --max-layer-sparsity 0.25
```

### Citation
Please give credits to this paper if this code is useful and helpful for your research.


      @inproceedings{huang2026gap,
      title     = {GAP: Gradient-guided Adaptive Pruning for Large Language Models},
      author    = {Huang, Chenyi and Xu, Li and Xiu, Xianchao},
      booktitle = {Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
      pages     = { },
      year      = {2026},
      organization = {Springer}
     }

