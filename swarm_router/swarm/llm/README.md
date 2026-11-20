# swarm/llm — Phase 2 (real LLMs)

Task environment, providers, response cache and budget guard. A real run is one command.
By default every script runs in `--dry-run` mode on `SimProvider` and makes no network calls.

## Layout

| file | role |
|---|---|
| `config.py` | YAML → `LLMConfig` / `ModelSpec` (model ids, prices per 1M tokens, `api_key_env`, `sim:` block) |
| `providers.py` | `AnthropicProvider` (`anthropic` SDK if installed, otherwise urllib), `OpenAICompatProvider` (`/chat/completions`), `SimProvider` (dry-run) |
| `client.py` | `LLMClient.call()`: cache lookup → budget check → provider → charge → cache put → log |
| `cache.py` | sqlite cache under `.cache/llm/`; key = sha256(provider identity incl. model id/params, system, messages, max_tokens, sample_id) |
| `budget.py` | `Budget` (hard USD limit, raises `BudgetExceeded`), `CallLog` (jsonl, one line per call) |
| `embed.py` | task embedding x ∈ R^D: hashed n-grams (offline) or an OpenAI-compatible `/embeddings` endpoint; fixed random projection to D, L2-normalised |
| `roles.py` | role system prompts, team protocol plan → execute → verify (≤ 1 revision) |
| `tasks/` | ~20 hand-written items each: math (`math.jsonl`), code with asserts (`code.jsonl`), text classification/extraction (`text.jsonl`); loaders for GSM8K/MBPP jsonl; checkers; subprocess sandbox |
| `estimate.py` | offline USD estimate (calls × prompt tokens × prices) |
| `../envs/llm.py` | `LLMEnv`: the `Swarm` env interface backed by real team executions |

Experiments: `experiments/llm_run.py` (main comparison), `experiments/llm_context_degradation.py`,
`experiments/llm_role_spread.py`. Each one takes `--dry-run` (default), `--real`, `--estimate`,
`--budget-usd`, `--n-tasks`, `--seeds`, `--config`.

## Running a real experiment

1. Copy `configs/llm.example.yaml` and check the model ids and prices (the prices in the file are
   first-party Anthropic list prices as of 2026-11; they are the only prices the code uses).
2. Export the key under the name given in `api_key_env`, e.g. `export ANTHROPIC_API_KEY=...`.
   Optionally `pip install anthropic`; without it the client uses raw HTTPS.
3. For more than 62 distinct items, put GSM8K/MBPP-format files into `data/`
   (`data/gsm8k*.jsonl`, `data/mbpp*.jsonl`, `data/text*.jsonl`). With only the built-in pool,
   `--n-tasks` > 62 repeats items, and with the cache a repeated item returns the same response.
4. Look at the estimate first, measure real output lengths with a small pilot, then run:
   ```bash
   python3 experiments/llm_run.py --estimate --n-tasks 300 --seeds 5
   python3 experiments/llm_run.py --real --budget-usd 0.5 --pilot 6 --pilot-only     # 18 calls, re-estimates
   python3 experiments/llm_run.py --real --budget-usd 5 --n-tasks 300 --seeds 5
   ```
   The estimate has two columns. **optimistic** uses `sim.out_tokens`. **conservative** uses
   `max_tokens` for models with `thinking: true`: adaptive thinking is billed as output, and nothing
   bounds it except max_tokens. That is also the per-call worst case the guard assumes. `--pilot N`
   makes N calls per model (round-robin executor/planner/verifier), records the real output tokens
   and prints the estimate again with them (optimistic = measured mean, conservative = measured max).
5. Spend gate for `--real`: you must state the budget explicitly, either with `--budget-usd X` on the
   command line or with `SWARM_ALLOW_SPEND_USD` set to at least X. You must also either type `yes` at
   the prompt or pass `--yes`. `--yes` alone is not enough. Without a TTY and without both, the
   script aborts before any call.
6. Results go to `results/<name>.json` (experiment name, git commit, argv, timestamp, full config,
   SwarmConfig, seeds, per-seed and mean ± sd metrics, budget). Every call is logged to
   `results/llm_calls/<name>_<time>.jsonl` with model, tokens, cost and whether it was a cache hit.

Suggested order (CLAUDE.md §4–5): `llm_role_spread.py` (is there a role × model spread at all?) and
`llm_context_degradation.py` (is there a context-length effect?) before building on either, and a
hypothesis entry in `notes/hypotheses.md` before every run.

## Budget guard

- Before every call that is not served from the cache:
  `estimate = ceil(chars/4) · price_in + max_tokens · price_out`. If `spent + estimate > limit`, the
  client raises `BudgetExceeded` and the call is **not** sent. Output is bounded by `max_tokens`, so
  real spend stays under the limit, up to the error of the chars/4 estimate for input tokens.
- After the call, the actual `usage` from the response is charged
  (input + 1.25·cache_write + 0.1·cache_read at the input price, plus output at the output price).
- Cache hits cost 0. Embedding calls (non-hashed provider) go through the same guard.
- **Per-variant shares** (`run.variant_budget_share`). `equal` (the default) gives each variant
  limit / n_variants. `estimate` makes shares proportional to each variant's conservative estimate.
  A mapping sets the fractions per variant label. A variant that exhausts its share stops with
  `stopped_budget[share]` and the next variant still runs. Exhausting the run limit
  (`stopped_budget[run]`) stops everything. Variants run in the order swarm, linucb, singles,
  pipeline, so the swarm is never the one starved. The pilot and the embeddings are charged to the
  run budget before the shares are computed.
- Any stop (budget or an exception in a seed) keeps the partial seed. The results file is always
  written, with the status per seed, per variant and per run (`ok`, `partial`, `stopped_budget`,
  `error`). A variant with no complete seed is reported from its partial seed and marked `(partial)`.
- HTTP calls are retried on 408/409/429/5xx/529 and on connection errors and timeouts: 4 attempts,
  exponential backoff of 1, 2 and 4 s, or `retry-after` if the server sends it. The SDK uses
  `max_retries=3`.
- A real embeddings provider without a positive `price_in_per_1m` is a config error (`ValueError`),
  so its calls can never be treated as free. Embedding cost is a separate row of the estimate.
- The guard also runs in dry-run, against simulated costs, so a dry run shows whether a configuration
  would stop early.

## How real success gets to the unchanged Swarm

`Swarm.step` computes `succ = u < env.p_success(t, m, members) · (1 − 0.3·fatigue)`. `LLMEnv` sets
`U[t] = 0.5`, and `p_success` actually runs the team on task t with model m (roles by team size:
1 → executor; 2 → executor + verifier; 3 → planner + executor + verifier; all members use the same m),
runs the checker and returns 1.0 or 0.0. So `u < p` reproduces the checker verdict. `reward()`, which
is called right after for the same step, returns `succ − lam · usd / cost_norm_usd` using that
execution's cost. `env.cost[m]` is the normalised expected cost of one executor call. Baselines call
`env.execute()`, which is the same code path.

## Cost accounting in rewards vs budget

Rewards use the **nominal** cost of each response, which is stored in the cache together with the
text. A cached response therefore gives the same reward as a fresh one, and results do not depend on
the cache state. The budget is charged only for calls actually made.
`include_seed_in_cache_key: true` (the default, used for the headline llm_run comparison) means one
model sample per seed, so the sd over seeds includes the model's sampling noise. Within a seed,
variants share responses to identical prompts (common random numbers). The "cache-aware cap" in
`--estimate` is then "each item × model × team shape paid once per seed". `false` shares responses
across seeds as well, which is cheaper, but then seeds vary only the task order and the router.

The results table reports reward ± sd over seeds and also the **SEM over items**: per-item mean
reward over all its occurrences, sd/√n_items. With 62 built-in items this ceiling usually dominates.
It counts calls stopped at max_tokens (truncated), refusals, and verifier replies with no VERDICT
line. "No verdict" takes the default action (accept, no revision) but is counted separately.

## Dry run (SimProvider)

`SimProvider` reads the `sim:` block of each model: accuracy per task kind, role skill, plan bonus,
verifier detect and false-reject rates, context decay per 1k input tokens, and output tokens per role.
It finds the task between the `<task>` tags, decides correctness with a hash of the request, and
writes a correct or a wrong answer in the real output format. That answer then goes through the
real parser, checker and sandbox, and the cost through the real prices. These numbers are made up.
A dry run checks the pipeline only and tells you nothing about the real models.

## Limitations

- The code sandbox writes the candidate into a fresh temp dir and uses it as cwd, HOME and TMPDIR.
  It runs `python -I`, enforces a timeout with a process-group kill and a bounded wait for the pipes,
  and sets rlimits on CPU, memory, file size, open files and process count. It is **not** a security
  boundary. There is **no filesystem isolation**: the candidate can read and write anything the user
  can. There is no network or syscall filtering, and RLIMIT_NPROC is per user and ignored for root.
  Run real experiments inside a container.
- Thinking tokens on current Anthropic models are billed as output. `effort: low` keeps them small,
  but only `max_tokens` bounds them. Budget against the conservative column, or run `--pilot` first.
- The verifier probes in `llm_role_spread.py` are plausible wrong answers with known labels. For
  code they are AST mutants of the reference (flipped comparison, swapped branch, off-by-one
  constant, dropped `abs`/`*`), checked to fail the tests. For math they are off-by-one values, and
  for text another label or a distractor entity from the text.
- With the built-in 62-item pool, a 300-task run repeats every item about 5 times. Use `data/` for
  real conclusions.
- The server-side refusal `fallbacks` parameter is deliberately not set: a fallback model would
  silently change the arm the router is learning about. A refusal yields empty text, which counts as
  a failure, and is visible as `stop_reason` in the call log.
