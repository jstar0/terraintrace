# Architecture

```text
fixture.py -> grouped TerrainDataset -> TerrainCase / low_res
                                      |
                         nearest + bilinear baselines
                                      |
                         fit_spectral_prior(train only)
                                      |
                    spectral reconstruction + area projection
                                      |
                         metrics.py -> per-case evidence
                                      |
              benchmark.py / sweep.py -> stable public artifacts
```

`model.py` is the array boundary. It validates finite two-dimensional fields,
scale divisibility, and immutable case metadata; its resampling primitives are
independent of reporting and optimization. `fixture.py` owns only invented
analytic surfaces and a group-held-out split.

`methods.py` contains the two interpolation baselines and the train-only radial
spectral prior. The prior is deliberately bounded and observation-consistent;
it cannot access validation or test targets. `metrics.py` keeps elevation,
gradient, edge, and low-resolution consistency measures in one typed record so
a report cannot silently switch protocols.

`benchmark.py` evaluates validation and test cases in a stable method order and
serializes no raster arrays. `sweep.py` reruns that contract across explicit
seeds. Both writers hash every public artifact after it has been written.

The implementation has no server, network fetch, database, model checkpoint,
or framework runtime. That makes split leakage, numerical drift, and artifact
changes easy to inspect in a clean checkout.
