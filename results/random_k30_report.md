# seeded random campaign

seed `20261021`, k=30, 14 default layers, all eight bits, 200 fixed images.
3,360 trials, 672,000 trial-image evaluations, 3,322 distinct physical faults.

## checks and repeatability

independently recomputed every cell seed and element through the python schema,
checked original/xor bytes, trial counts, outcome totals and accuracy/rate consistency.
verified every subset payload against the manifest and the model against its recorded hash.
the separate k=1 pilot repeats the first draw in every layer-bit cell: all 112
rows agree in every field except the sequential trial id. this includes seeds,
selected weights and all outcomes. the runner also repeats its clean baseline
and checks full-buffer restoration after every fault. the full 3,360 rows
were not executed twice.

model sha256: `780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1`.

subset hash (sha256 of manifest.json): `9c6d5d1e0e20c551cdcb01d9894a58743bc377b9fa4cb82a1750cd0c3dedea7a`.

campaign csv sha256: `4fb1ca43c28802f9a20484b4f6af100be279a54452549812dce68c2d10964420`.

## measures and denominators

one trial flips one stored weight bit, keeps it flipped for all 200 images,
then restores it. masked means an identical int8 output vector; subtle means
changed vector with unchanged top-1; catastrophic means changed top-1, including
wrong-to-right and wrong-to-wrong. due records returned c-api errors only.
top-1 mismatch is (catastrophic + due) / trial-image evaluations. accuracy
drop is clean accuracy minus faulted accuracy, in percentage points, with
returned errors counted incorrect. clean accuracy is 130/200 = 65%.

outcome counts over 672,000 trial-image evaluations: `{"n_catastrophic": 35723, "n_due": 0, "n_masked": 138230, "n_subtle": 498047}`.

changed output vectors: 533770/672000 trial-image evaluations.
trials with any changed output vector: 3072/3360.
trials with any changed top-1 or returned error: 2604/3360.

## conditional summaries

entries are mean [approximate 95% monte carlo interval]. mismatch is percent;
accuracy drop is percentage points. intervals use 1.96 times the stratified
standard error of trial-level rates, with sample variance estimated separately
in each layer-bit cell. images are fixed and reused, never independent samples.
each interval is marginal; comparisons across many groups need separate
multiplicity handling. these intervals describe uncertainty from weight
draws under this pseudorandom design, conditional on these images, model and library. k=30 is small for rare
large effects; normal intervals may under-cover and can cross physical bounds.
zero observed effects cannot rule out rare effects.

† when every trial in a group is zero for a measure, the interval instead
allows unseen rare effects. q = 1 - 0.05^(1/n) is a conservative 95% upper
bound on the equal-weight probability of a nonzero trial; multiply q by
the measure bounds [0, 1] for mismatch or [-0.35, 0.65] for accuracy drop.
independent bounded draws in equal-sized strata support this bound even
when stratum event probabilities differ: the probability of no events is
at most (1 - mean event probability)^n. this does not assume image independence.
other zero-variance cases use a conservative bounded-variable hoeffding
interval instead of a zero-width normal interval.

the sampler draws with replacement via splitmix64 modulo the layer size,
with negligible modulo imbalance. repeats are retained. seeds are deterministic,
so probabilistic interpretation assumes the seeded generator approximates
independent uniform draws. summaries weight selected layers and bits equally,
not in proportion to weight counts. they do not estimate a uniform fault
probability over the whole model, other layers, other images or hardware.

| group | trials | trial-image evaluations | top-1 mismatch % | accuracy drop pp |
|---|---:|---:|---:|---:|
| all selected cells | 3360 | 672000 | 5.316 [5.090, 5.542] | 0.819 [0.682, 0.956] |
| bit 0 | 420 | 84000 | 2.532 [2.410, 2.654] | -0.256 [-0.309, -0.203] |
| bit 1 | 420 | 84000 | 2.654 [2.519, 2.789] | -0.185 [-0.242, -0.127] |
| bit 2 | 420 | 84000 | 2.800 [2.641, 2.959] | -0.195 [-0.253, -0.138] |
| bit 3 | 420 | 84000 | 3.285 [3.043, 3.526] | -0.071 [-0.147, 0.004] |
| bit 4 | 420 | 84000 | 4.362 [3.918, 4.806] | 0.185 [-0.019, 0.388] |
| bit 5 | 420 | 84000 | 6.240 [5.526, 6.955] | 1.021 [0.582, 1.461] |
| bit 6 | 420 | 84000 | 8.945 [7.890, 10.000] | 2.454 [1.788, 3.119] |
| bit 7 | 420 | 84000 | 11.710 [10.554, 12.865] | 3.598 [2.885, 4.310] |
| op 1 | 240 | 48000 | 13.756 [12.444, 15.069] | 3.102 [2.394, 3.810] |
| op 3 | 240 | 48000 | 22.560 [20.031, 25.090] | 8.977 [7.314, 10.641] |
| op 9 | 240 | 48000 | 8.815 [7.971, 9.658] | 0.190 [-0.241, 0.620] |
| op 22 | 240 | 48000 | 5.196 [4.835, 5.557] | -0.381 [-0.528, -0.234] |
| op 36 | 240 | 48000 | 3.717 [3.529, 3.904] | -0.188 [-0.272, -0.103] |
| op 42 | 240 | 48000 | 6.427 [5.564, 7.291] | 0.579 [0.181, 0.977] |
| op 56 | 240 | 48000 | 3.348 [3.171, 3.525] | -0.081 [-0.162, -0.001] |
| op 89 | 240 | 48000 | 1.775 [1.490, 2.060] | -0.071 [-0.170, 0.029] |
| op 60 | 240 | 48000 | 2.427 [2.260, 2.595] | -0.075 [-0.142, -0.008] |
| op 62 | 240 | 48000 | 2.760 [2.528, 2.993] | -0.121 [-0.198, -0.044] |
| op 64 | 240 | 48000 | 2.385 [2.195, 2.576] | -0.146 [-0.223, -0.069] |
| op 102 | 240 | 48000 | 1.058 [0.848, 1.269] | -0.215 [-0.272, -0.158] |
| op 116 | 240 | 48000 | 0.194 [0.154, 0.234] | -0.110 [-0.141, -0.080] |
| op 120 | 240 | 48000 | 0.004 [-0.002, 0.010] | 0.002 [-0.002, 0.006] |

lowest observed faulted accuracy: 0.005000 on trial 467,
op 3, element 63, bit 7.

## different from the corner campaign

the historical campaign used five deterministic rules per layer-bit cell:
560 rule-weighted trials, 528 distinct faults. this campaign uses 30 seeded
element draws per cell. its means and uncertainty belong to that different
design. neither design makes the 200 selected images representative of imagenet.
zero returned errors is not evidence of crash/hang detection. all 55 weight
tensors have per-channel scale metadata; the inventory retains only scale 0.
no injector logic, input preprocessing or historical results were changed.

## runtime and run record

the initial estimate used the first 40 completed pilot trials: 37.3 seconds,
projecting 3,133.2 seconds (52.2 minutes) for 3,360 trials. this was reported
before launching k=30. the full k=1 pilot across all 112 cells then completed
in 105.2 seconds, consistent with a 3,156-second projection (52.6 minutes).
k=30 stayed within the approximately one-hour threshold. the full runner loop took 3136.9 seconds
(52.28 minutes), excluding model loading and the two clean baseline
passes. both processes exited successfully and logged a clean final model buffer.

see `random_k30_provenance.json` for the exact original command, cwd, seed,
k, model/subset/library hashes, reported c library version, compiler, source
hashes and python versions. `random_k30.log` records progress and completion.
the campaign used the original installed c library, reported version 2.19.0,
sha256 `2eb9b9b2b108a4af52e91deafc05590b3e9672e7b3775c894d761d2b615e0a16`. its source revision and build flags remain
unverified. the separate fresh source build in the setup records was not used
for this campaign.

## pooled accuracy and classifier observations

faulted accuracy pooled across all 3,360 trials is 431,298/672,000 = 64.18125%.
the clean reference is 130/200 = 65%, reused for each trial. their difference
is 0.81875 percentage points, equivalent to 5,502 fewer correct predictions
over 672,000 trial-image evaluations. this reuse does not add independent
clean images or justify imagenet population intervals.

op 120 had two top-1 changes in 48,000 trial-image evaluations across 240
random trials. trials 3305 and 3315 each changed one image prediction, both
at bit 6, elements 37531 and 113596. the first trial got 129/200 correct;
the second retained 130/200. the historical corner result of zero changes
across 40 faults remains true for its own design and does not generalize to
these random faults. op 3 and bit 7 have the highest observed group means
in this random design under both mismatch and accuracy-drop measures.

the run started at `2026-10-21T23:08:05.643029-04:00` and finished at
`2026-10-22T00:00:22.500161-04:00` in america/new_york, crossing local midnight.
