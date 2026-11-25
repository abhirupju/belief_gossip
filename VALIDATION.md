# Validation report: paper, protocol, and calibration

Reviewed on 25 November 2026 against commit `104722c224f69c94b28f5f945a6678298960db9c` and the local [paper draft](belief_gossip.pdf), “Gossiping Belief among Small Language Model Powered Personal Agents.”

This report documents the three concerns identified in the review. It makes no changes to source code, tests, prompts, or configuration. Suggested changes and acceptance checks below are recommendations for subsequent work.

Source links use repository-relative paths with `#L` line anchors for repository browsers. Function names and line numbers are included in the link text for editors that do not support those anchors. Line numbers refer to the reviewed commit.

## Assessment and scope

The core HiddenBench implementation appears consistent with the draft's one-round construction: option unrotation, normalized geometric averaging over rotations, posterior-minus-prior lift, margin scoring, joint calibration across tasks, and the finite-sample conformal quantile. The review did not identify a major defect in those operations.

The concerns have different meanings and should not be conflated:

| Concern | Classification | Relevance to the current draft | What is established |
| --- | --- | --- | --- |
| 1. Missing hub-fallback accuracy calculation | Reproducibility and evaluation gap | Direct: Algorithm 2 and Tables 1 and 3 | The checked-in protocol evaluator does not compute the reported final-answer accuracy. |
| 2. Default re-scoring prior differs from the paper | Configuration and algorithm alignment gap | Direct: Equation (15) and the multi-round results | The default and paper settings can produce different final answers on the same cache. |
| 3. Question-level calibration after user-level splitting | Statistical validity problem | Separate: WearableQA is not evaluated in this draft | User separation alone does not justify the advertised coverage guarantee; a runnable counterexample gives 66.7% at a 90% target. |

The first two findings do **not** establish that the paper's numerical results are wrong. A separate analysis script and the correct configuration could explain them. The experiment predictions, re-scoring caches, and table-generation script were not present in this checkout, so their provenance and numerical outputs could not be verified.

All numerical examples below are explicitly synthetic unless described as values reported in the draft. They are diagnostic examples, not estimates of the error in the real experiments.

## 1. The checked-in evaluator does not calculate the paper's final-answer accuracy

### 1.1 What the paper specifies

Algorithm 2, page 5, returns both a set and an answer:

1. For a singleton, return its only candidate.
2. For a nonempty set with several candidates, return the hub's most probable candidate **within that set**, using its original posterior `p(y | Q, S, P_hub)`.
3. For an empty set, return no answer; evaluation counts this as an error.

Section 6.1, pages 6–7, says accuracy is averaged over the choice of agent acting as hub. Table 3 reports, for example, 76.4% for the 2B model with two rounds and 83.2% for the 4B model with two rounds. These are final-answer accuracies, not coverage or accuracy conditional on producing a singleton.

### 1.2 What the code computes

| Source | Observed behavior |
| --- | --- |
| [`run_protocol`, protocol.py:133](src/analysis/protocol.py#L133) | Computes intersections, stopping conditions, and communication counters. |
| [`run_protocol` return, protocol.py:164](src/analysis/protocol.py#L164) | Returns `final_set`, trajectories, and counters; no hub answer or per-hub correctness. |
| [`evaluate_protocol`, protocol.py:253](src/analysis/protocol.py#L253) | Calls `evaluate_sets`, then adds round, bit, model-pass, and first-miss metrics. It does not implement the hub fallback. |
| [`evaluate_sets`, conformal.py:86](src/analysis/conformal.py#L86) | Reports coverage, set size, singleton rate, singleton accuracy, and empty rate. |
| [`run_multiround` output, protocol.py:351](src/analysis/protocol.py#L351) | Saves aggregate metrics and trajectories for split 0 only. |
| [`agent_view`, scores.py:68](src/analysis/scores.py#L68) | Already supplies each agent's rotation-averaged original posterior, which is sufficient for the missing decision rule. |

The missing calculation matters because none of the existing set metrics is interchangeable with final-answer accuracy. Also, split-0 trajectories alone cannot reconstruct the mean final-answer accuracy across all 200 splits. The evaluator would need to run the fallback on each split, or the separate table-generation script would need to do so.

### 1.3 Worked example: four different metrics from the same three tasks

Suppose three tasks have two possible hubs each. The final sets and hub decisions are:

| Task | Truth | Final set | Answer if hub 0 | Answer if hub 1 | Accuracy averaged over hubs |
| --- | --- | --- | --- | --- | --- |
| 1 | A | `{A}` | A | A | 1.0 |
| 2 | A | `{A, B}` | B | A | 0.5 |
| 3 | B | `{}` | None | None | 0.0 |

For task 2, let the original posterior probabilities over `(A, B, C)` be:

- Hub 0: `(0.2, 0.6, 0.2)`, so the fallback picks B.
- Hub 1: `(0.7, 0.2, 0.1)`, so the fallback picks A.

The resulting metrics are:

| Metric | Calculation | Value |
| --- | --- | --- |
| Coverage | Truth belongs to the set on tasks 1 and 2 | 2/3 = 66.7% |
| Singleton/decided rate | Only task 1 has a singleton | 1/3 = 33.3% |
| Singleton accuracy | The only singleton is correct | 1/1 = 100% |
| Paper's final-answer accuracy | `(1.0 + 0.5 + 0.0) / 3` | 50% |

Reporting any of the first three as the fourth would change the scientific conclusion. The current code does **not** appear to make that substitution; it simply does not provide the fourth metric.

### 1.4 Runnable reproduction using the existing evaluator

Run from the repository root in an environment with the project's lightweight dependencies. The commands in this report use synthetic data, do not load a model, and do not write experiment outputs. `PYTHONDONTWRITEBYTECODE=1` also prevents Python bytecode writes.

The example uses a common uniform prior and posterior distributions whose margins generate the sets above at threshold 1.3. The local `paper_task_accuracy` function demonstrates the missing evaluation step; it is not an existing repository function.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
import math
from analysis.protocol import evaluate_protocol
from analysis.scores import argmax

def agent(index, probabilities):
    posterior = {y: math.log(p) for y, p in zip("ABC", probabilities)}
    prior = {y: -math.log(3) for y in "ABC"}
    return {
        "agent_index": index,
        "prior": prior,
        "posterior": posterior,
        "lift": {y: posterior[y] - prior[y] for y in "ABC"},
    }

def task(index, truth, agents):
    return {
        "task_index": index,
        "true_letter": truth,
        "letters": list("ABC"),
        "agents": agents,
    }

views = [
    task(0, "A", [
        agent(0, [0.8, 0.1, 0.1]),
        agent(1, [0.8, 0.1, 0.1]),
    ]),
    task(1, "A", [
        agent(0, [0.2, 0.6, 0.2]),
        agent(1, [0.7, 0.2, 0.1]),
    ]),
    task(2, "B", [
        agent(0, [0.8, 0.1, 0.1]),
        agent(1, [0.1, 0.8, 0.1]),
    ]),
]
metrics, trajectories = evaluate_protocol(views, "margin", [1.3])

def paper_task_accuracy(view, kept):
    if not kept:
        return 0.0
    answers = [
        argmax({y: a["posterior"][y] for y in kept})
        for a in view["agents"]
    ]
    return sum(y == view["true_letter"] for y in answers) / len(answers)

accuracy = sum(
    paper_task_accuracy(v, t["final_set"])
    for v, t in zip(views, trajectories)
) / len(views)
print("final sets:", [t["final_set"] for t in trajectories])
for key in ["coverage", "singleton_rate", "singleton_accuracy"]:
    print(f"{key}: {metrics[key]:.6f}")
print(f"paper final-answer accuracy: {accuracy:.6f}")
print("final-answer accuracy returned by evaluator:", "accuracy" in metrics)
PY
```

Expected output:

```text
final sets: [['A'], ['A', 'B'], []]
coverage: 0.666667
singleton_rate: 0.333333
singleton_accuracy: 1.000000
paper final-answer accuracy: 0.500000
final-answer accuracy returned by evaluator: False
```

### 1.5 Correct aggregation and checks for the author

For task `j` with `N_j` agents and final set `C_j`, define:

```text
a_j = 0                                                    if C_j is empty
a_j = (1/N_j) * sum_i 1[argmax_{y in C_j} q_original_ji(y) = truth_j]
                                                            otherwise
split_accuracy = mean_j(a_j) over that split's test tasks
reported_accuracy = mean_s(split_accuracy_s) over the 200 splits
```

Use the original posterior specified by Algorithm 2, not the re-scored final-round posterior. Restrict the argmax to the surviving set. Renormalization within the set is unnecessary for argmax. Specify the tie policy; the existing [`argmax`, scores.py:58](src/analysis/scores.py#L58) breaks ties by letter order.

Averaging over hubs should happen **inside each task**. For example, if a three-agent task has no correct hub decisions and a four-agent task has four correct hub decisions, equal task weighting gives `(0/3 + 4/4)/2 = 50%`. Pooling all hub decisions gives `4/7 = 57.1%`, a different estimand. This is an illustrative weighting risk, not a confirmed error in the unavailable table script.

To close this finding:

1. Locate and include the script that produced Tables 1 and 3, or implement the specified metric in a later change.
2. Verify singleton, non-singleton, empty-set, posterior-tie, and unequal-agent-count cases.
3. Reproduce the per-hub, per-task, per-split values before rounding the table entries.
4. Retain enough intermediate data to reproduce all 200 splits; the current split-0 trace is useful for inspection but insufficient by itself.
5. Confirm the reported 2B/4B two-round accuracies of 76.4%/83.2% from the actual outputs. This review has not independently reproduced those numbers.

For a fixed test split, final-answer accuracy must not exceed set coverage: the fallback is restricted to the final set. Correct singletons contribute to both metrics. These are useful consistency checks, but they do not replace the missing calculation.

## 2. The default multi-round prior differs from Equation (15)

### 2.1 The two algorithms

Let `q_t(y)` be the posterior after an agent reads its private fact and the current group conclusion. Let `p_0(y)` be its original shared-context prior and `p_t(y)` the prior re-scored with the group conclusion but without the private fact.

The draft's Equation (15), page 5, uses:

```text
paper:   lift_t(y) = log q_t(y) - log p_0(y)
```

The default variant-D simulation instead uses:

```text
default: lift_t(y) = log q_t(y) - log p_t(y)
```

Relevant code:

- [`--variant`, simulate_protocol.py:23](src/simulate_protocol.py#L23) defaults to `A`, the frozen-score simulation. The paper's re-prompting algorithm requires `D`.
- [`--lift_prior`, simulate_protocol.py:26](src/simulate_protocol.py#L26) defaults to `conclusion`, even after selecting `D`.
- [`make_rescore`, simulate_protocol.py:83](src/simulate_protocol.py#L83) selects the fixed-prior wrapper only for `--lift_prior shared`.
- [`shared_prior_rescore`, protocol.py:59](src/analysis/protocol.py#L59) correctly takes the prior from the empty ruling and the posterior from the current ruling.
- [`rescored_agent`, protocol.py:116](src/analysis/protocol.py#L116) subtracts whichever prior that lookup returns.

The difference between the lifts is `log p_t(y) - log p_0(y)`, which generally depends on the candidate. It therefore survives the margin calculation. In contrast, renormalizing a distribution over the surviving set changes all its log probabilities by the same constant, which cancels from margins. These are different operations.

### 2.2 Numerical example: the same cache produces opposite answers

Use three candidates `(A, B, C)`, one agent, two rounds, and the same threshold `tau = 0.5` in each round. For simplicity, specify distributions using logits, meaning probabilities are obtained by softmax.

| Distribution | Logits over `(A, B, C)` | Approximate probabilities |
| --- | --- | --- |
| Original prior `p_0` | `(0, 0, 0)` | `(0.3333, 0.3333, 0.3333)` |
| Original posterior `q_0` | `(0, 0, -3)` | `(0.4879, 0.4879, 0.0243)` |
| Prior after C is ruled out, `p_1` | `(0, 3, 0)` | `(0.0453, 0.9094, 0.0453)` |
| Posterior after C is ruled out, `q_1` | `(0, 2, -3)` | `(0.1185, 0.8756, 0.0059)` |

In round 0, both methods have margins `(0, 0, 3)`. They exclude C and retain `{A, B}`.

In round 1:

| Prior choice | Lifts over `(A, B)`, ignoring a common offset | Margins over `(A, B)` | Final set at `tau = 0.5` |
| --- | --- | --- | --- |
| Paper: fixed `p_0` | `(0, 2)` | `(2, 0)` | `{B}` |
| Default: conclusion-conditioned `p_1` | `(0, -1)` | `(0, 1)` | `{A}` |

If A is the truth, the default covers it and the paper setting does not. If B is the truth, the direction reverses. This example establishes that the setting changes the algorithm; it does not establish that either setting performs better on HiddenBench.

### 2.3 Runnable reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
from analysis.protocol import run_protocol, shared_prior_rescore
from analysis.scores import normalize

def log_distribution(logits):
    return normalize(dict(zip("ABC", logits)))

def cache(task_index, agent_index, ruling):
    assert ruling in ("", "C")
    return {
        "prior": log_distribution([0, 0, 0] if not ruling else [0, 3, 0]),
        "posterior": log_distribution([0, 0, -3] if not ruling else [0, 2, -3]),
    }

view = {
    "task_index": 0,
    "true_letter": "A",
    "letters": list("ABC"),
    "agents": [{"agent_index": 0}],
}
for name, lookup in [
    ("default conclusion prior", cache),
    ("paper shared prior", shared_prior_rescore(cache)),
]:
    trajectory = run_protocol(view, "margin", [0.5, 0.5], "D", lookup)
    print(name, "->", trajectory["sets"])
PY
```

Expected output:

```text
default conclusion prior -> [['A', 'B'], ['A']]
paper shared prior -> [['A', 'B'], ['B']]
```

### 2.4 Configuration that matches the paper

For an existing, matching pair of predictions and cache files, the following selects the draft's margin-based protocol and its stated split sizes. Replace the two paths with actual artifacts. `--no_save` avoids overwriting a saved report.

```bash
python src/simulate_protocol.py \
  --predictions /path/to/predictions.json \
  --rescored /path/to/rescored.json \
  --variant D \
  --lift_prior shared \
  --score margin \
  --schedule global \
  --rounds 0 1 2 \
  --alpha 0.1 \
  --cal_fraction 0.5 \
  --splits 200 \
  --seed 0 \
  --no_save
```

The seed above is the code default; confirm the seed actually used for the paper rather than assuming it. This command generates the existing set/protocol metrics, **not** the missing final-answer accuracy from concern 1.

The indexing distinction is essential:

| Paper maximum total rounds | CLI `--rounds` value | Internal round indices |
| --- | --- | --- |
| `T = 1` | `0` | `0` |
| `T = 2` | `1` | `0, 1` |
| `T = 3` | `2` | `0, 1, 2` |

See [`--rounds`, simulate_protocol.py:31](src/simulate_protocol.py#L31) and [`calibrate_global`, protocol.py:192](src/analysis/protocol.py#L192).

The empty-ruling cache should use the same `answer_only` prompt as the one-shot predictions, without an added group-conclusion sentence. This behavior exists in [`template_for_ruling`, belief_update.py:86](src/belief_update.py#L86) and is selected by the default [`--round0_template`, belief_update.py:157](src/belief_update.py#L157). Verify the saved template and rotation metadata, not just the cache filename.

### 2.5 Checks for the author

1. Recover the actual command/configuration for each reported row, including variant, lift prior, threshold schedule, extra-round count, calibration fraction, seed, model, and prompt versions.
2. Check that the saved report has `variant: D`, `lift_prior: shared`, and the intended `schedule: global` rows. The report includes the prior setting at [`simulate_file`, simulate_protocol.py:101](src/simulate_protocol.py#L101).
3. Confirm that predictions and cache refer to the same tasks, agent fact partitions, model, candidate orders, and prompt text. Compare normalized empty-ruling cached scores with the prediction-derived scores using a justified numerical tolerance.
4. On the same splits, compare `shared` and `conclusion`, retaining per-task trajectories. Count final-set changes, newly decided tasks, and truth losses. Do not infer equivalence from similar aggregate coverage alone.
5. Recompute the paper's delta as the fraction of **all test tasks** whose truth survived round 0 and was excluded later. It is not the conditional miss rate among round-0 survivors.

For a fixed run with nested sets and identical test tasks, delta equals `coverage_round0 - coverage_final`. The existing [`missed_round_r` metrics, protocol.py:268](src/analysis/protocol.py#L268) can also be summed over `r >= 1`. Compare full-precision values; differences between rounded table entries need not equal the rounded delta.

**Not a separate finding:** Section 5.3 of the draft explicitly acknowledges that reusing a round-0 threshold can lose additional coverage, and writes the bound as `1 - alpha - delta`. The review therefore does not claim that this choice contradicts the paper. The concern is whether the reported results came from the stated algorithm.

## 3. WearableQA separates users but calibrates as if questions were exchangeable

### 3.1 Where the mismatch occurs

WearableQA correctly prevents a user's questions from appearing in both calibration and test. However, the threshold is then calculated from all calibration **questions**, with sample size equal to the number of questions. User-level exchangeability does not imply question-level exchangeability when a user's questions may be dependent.

| Source | Role in the issue |
| --- | --- |
| [`make_group_splits`, analyze_wqa.py:191](src/analyze_wqa.py#L191) | Selects half the distinct users for calibration. |
| [`split_views`, analyze_wqa.py:204](src/analyze_wqa.py#L204) | Expands selected users back into individual question views. |
| [`run_grouped_conformal`, analyze_wqa.py:249](src/analyze_wqa.py#L249) | Passes the question list to ordinary calibration. |
| [`calibrate`, conformal.py:70](src/analysis/conformal.py#L70) | Produces one joint score per view/question. |
| [`conformal_quantile`, conformal.py:46](src/analysis/conformal.py#L46) | Uses the number of those scores as `n`. The formula itself is correct for the appropriate exchangeable unit. |
| [`classwise_thresholds`, analyze_wqa.py:271](src/analyze_wqa.py#L271) | Filters questions by label, then applies the same calibration. |
| [`run_structured_conformal`, analyze_wqa.py:388](src/analyze_wqa.py#L388) | Uses the same question-level classwise thresholds for each axis at `alpha/2`. |

The guarantees advertised in [README section 11c, line 356](docs/README.md#L356) therefore need a different calibration argument or implementation. This concerns the standard joint, classwise, and structured paths, not merely the separately documented approximate pooled-agent `markov` rule.

### 3.2 Worked counterexample: 20 calibration rows contain only two user scores

Construct four users. Each has ten identical questions, all with truth A. Each question has one agent and four candidates. The posterior probability of A is constant within a user:

| User | Questions | `p(A)` | True-answer LAC score `1 - p(A)` |
| --- | --- | --- | --- |
| u1 | 10 | 0.6 | 0.4 |
| u2 | 10 | 0.7 | 0.3 |
| u3 | 10 | 0.8 | 0.2 |
| u4 | 10 | 0.9 | 0.1 |

Use `alpha = 0.1`, `k = 1`, the joint rule, fixed scores, and no empirical-prior estimation. Select two users for calibration and two for test, exactly matching the half-user design.

There are 20 calibration questions. The code uses:

```text
rank = ceil((20 + 1) * (1 - 0.1)) = ceil(18.9) = 19
```

Because each selected user's score is repeated ten times, the nineteenth score is the larger of the **two** distinct user scores. Enumerating every possible calibration-user pair gives:

| Calibration users | Threshold | Test users | Fraction of test questions covered |
| --- | --- | --- | --- |
| u1, u2 | 0.4 | u3, u4 | 1.0 |
| u1, u3 | 0.4 | u2, u4 | 1.0 |
| u1, u4 | 0.4 | u2, u3 | 1.0 |
| u2, u3 | 0.3 | u1, u4 | 0.5 |
| u2, u4 | 0.3 | u1, u3 | 0.5 |
| u3, u4 | 0.2 | u1, u2 | 0.0 |

Mean coverage is `(1 + 1 + 1 + 0.5 + 0.5 + 0) / 6 = 2/3 = 66.7%`, below the advertised 90% target. There is no user overlap, label fitting, prompt tuning, or model failure in this construction.

This is not only an unlucky finite-data split. Consider a population with independent continuous user scores, each copied into ten questions. With two calibration users, the code's threshold is their maximum. A new user's score is the largest of the three independent scores with probability `1/3`, so population coverage is `2/3`. The users are exchangeable; the expanded question rows do not satisfy the ordinary split-conformal premise.

With one score per calibration user, `n = 2` instead of 20. At `alpha = 0.1`, the conformal rank is `ceil(3 * 0.9) = 3 > 2`, so the threshold is infinite. Keeping all labels in this example reflects the actual lack of calibration information. Repeating a user's questions should not manufacture the information needed for a finite 90% threshold.

### 3.3 Runnable reproduction

This calls the current grouped and classwise evaluators. All four candidate probabilities are positive and sum to one. The classwise A threshold sees the same repeated scores because every question has truth A; absent classes receive infinite thresholds, which does not change true-label coverage.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
import itertools
import math
from analyze_wqa import run_grouped_conformal, run_classwise_conformal

views = []
for user, p_true in enumerate([0.6, 0.7, 0.8, 0.9], start=1):
    posterior = {
        "A": math.log(p_true),
        **{y: math.log((1.0 - p_true) / 3) for y in "BCD"},
    }
    prior = {y: -math.log(4) for y in "ABCD"}
    for question in range(10):
        index = (user - 1) * 10 + question
        views.append({
            "task_index": index,
            "task_id": f"q{index}",
            "user_id": f"u{user}",
            "gold_class": "CH",
            "true_letter": "A",
            "letters": list("ABCD"),
            "agents": [{
                "agent_index": 0,
                "prior": prior,
                "posterior": posterior,
                "lift": {y: posterior[y] - prior[y] for y in "ABCD"},
                "generated": [],
            }],
        })

splits = [set(pair) for pair in itertools.combinations(
    ["u1", "u2", "u3", "u4"], 2
)]
grouped = run_grouped_conformal(
    views, splits, score="lac", rule="joint", alpha=0.1,
    k=1, exclude_from_calibration=(), prior="prompt",
)
classwise = run_classwise_conformal(
    views, splits, score="lac", alpha=0.1, k=1, prior="prompt",
)
print("nominal coverage: 0.900000")
print(f"calibration questions: {grouped['n_calibration_mean']:.0f}")
print("calibration users: 2")
for name, result in [("grouped", grouped), ("classwise", classwise)]:
    coverage = result["metrics"]["coverage"]["mean"]
    print(f"{name} mean coverage: {coverage:.6f}")
PY
```

Expected output:

```text
nominal coverage: 0.900000
calibration questions: 20
calibration users: 2
grouped mean coverage: 0.666667
classwise mean coverage: 0.666667
```

The synthetic execution directly verifies the grouped and classwise paths. The structured path calls the same question-level threshold machinery; splitting alpha between axes does not establish its coverage unless each axis's calibration is itself valid.

### 3.4 Related issue: fitting the empirical prior on calibration observations

The default WearableQA analysis also changes the score function using its calibration inputs:

1. [`empirical_prior_views`, analyze_wqa.py:129](src/analyze_wqa.py#L129) averages posteriors from the selected calibration users.
2. The resulting prior is used to calculate lift/margin for those same calibration observations and the held-out observations.
3. Calibration proceeds on the observations that helped fit that prior at [`analyze_wqa.py:250`](src/analyze_wqa.py#L250); analogous calls occur in the classwise and structured methods.

The fact that fitting uses no labels does not by itself supply the ordinary split-conformal exchangeability argument. Calibration inputs help construct the score function, whereas test inputs do not. Conditional on the fitted prior, the usual “one fixed function applied to fresh exchangeable calibration and test examples” reasoning cannot simply be assumed.

For a concrete illustration, suppose two calibration posteriors assign probabilities 0.9 and 0.5 to A. Their empirical prior assigns A probability 0.7. The first observation's A lift is `log(0.9/0.7) = 0.2513`. If the reference prior had instead been fitted independently and assigned A probability 0.5, the same observation's lift would be `log(0.9/0.5) = 0.5878`. Its own contribution to the fitted prior changes its score. This illustrates the dependence; it is **not** a separate numerical proof of undercoverage for every lift/margin configuration. Some score constructions can cancel prior effects and need individual analysis. LAC does not use the prior, and the counterexample above deliberately uses LAC to isolate the user/question problem.

A separate prior-fitting user set, with the prior frozen before calibration, is a straightforward way to restore the required separation. A more data-efficient alternative needs its own justified calibration construction; ordinary cross-fitting should not be assumed to inherit exact split-conformal coverage automatically.

### 3.5 Checks and repair directions for the author

First specify the prediction target. These are distinct:

- A randomly sampled question from a new user.
- Every question in a specified batch from a new user.
- A question drawn with users weighted equally versus questions weighted equally.
- Coverage conditional on a phenotype or axis side.

Then match both calibration and evaluation to that target. Possible directions include:

| Target/design | Possible approach | Limitation to make explicit |
| --- | --- | --- |
| A question sampled from a new user | Select one question per calibration user using a fixed, label-independent sampling rule; use the corresponding test sampling rule. | Uses fewer observations; the guarantee is for that sampling scheme. |
| Simultaneous coverage over a new user's specified question batch | Calibrate one maximum true-answer score per calibration user/batch, under exchangeability of batches and a fixed scoring rule. | Can be conservative; classwise/axis-conditional claims still need their own derivation. |
| More efficient use of repeated questions | Use an appropriate hierarchical/group conformal construction with a stated guarantee. | Merely changing the rank formula's `n` to a user count while keeping all question scores is not such a construction. |
| Empirical prior for lift/margin | Fit on separate users and freeze before calibration. | Requires enough distinct fitting and calibration users; does not by itself fix question dependence. |

Acceptance checks for subsequent work:

1. Report both distinct-user counts and question counts, including per-class and per-axis counts.
2. Verify that no user crosses the intended fitting, calibration, and test partitions.
3. Re-run the duplicate-question counterexample. A user-maximum or one-question-per-user implementation should return an infinite threshold in this example at a 90% target.
4. For a chosen user-level design, check that duplicating an identical question within a user does not create extra calibration units or turn its infinite threshold finite.
5. Check sparse classes and axis sides using the actual exchangeable calibration units. Repeated questions are not a substitute for more users.
6. Re-evaluate the original WearableQA predictions after deciding the valid calibration design; distinguish measured empirical coverage from a proved coverage guarantee.

This finding does not invalidate the draft's HiddenBench results. That path calibrates on a single joint score per task, and the current PDF does not make a WearableQA experimental claim. The actual impact on the WearableQA numbers cannot be quantified without its tasks and saved predictions.

## Validation performed and remaining evidence

- Read the eight-page draft and visually inspected the algorithms, calibration argument, and experimental tables.
- Traced the HiddenBench prompt, score, rotation, cache, and protocol paths, plus the relevant WearableQA calibration paths.
- Executed the synthetic reproductions documented above against the reviewed source.
- Ran the existing non-real-model suite: **238 passed, 1 deselected**. The deselected test needs a real model. The suite emitted one warning about a sparse class in the small random-forest test fixture; it was not a test failure.
- Used a temporary environment for review dependencies. The original model weights and real experiment outputs were not available, so no real-model rerun or numerical table replication was performed.
- No major exploitable security vulnerability was identified in the source review; this was not a dependency vulnerability audit or a deployed-system penetration test.

The tests passing is compatible with these findings: the suite checks existing behavior, while concern 1 is missing evaluation behavior, concern 2 is a scientifically meaningful configuration choice, and concern 3 concerns the sampling assumptions behind a guarantee.

The most useful next evidence is the exact prediction/cache pair and the table-generation script for each model, together with the command/configuration and split seed used for the draft. Those artifacts would distinguish a release/documentation gap from a result that actually needs recomputation.
