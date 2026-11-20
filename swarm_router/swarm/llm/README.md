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
4. Look at the estimate first, then run:
   ```bash
   python3 experiments/llm_run.py --estimate --n-tasks 300 --seeds 5
   python3 experiments/llm_run.py --real --budget-usd 5 --n-tasks 300 --seeds 5
   ```
   `--real` prints the estimate again and asks you to type `yes` (or pass `--yes`). Without a TTY and
   without `--yes` it exits before making any call.
5. Results go to `results/<name>.json` (experiment name, git commit, argv, timestamp, full config,
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
- A run that hits the limit stops cleanly: the results file is still written with
  `status: stopped_budget` and the partial per-seed rows.
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
`include_seed_in_cache_key: false` (the default) means identical prompts share one response across
seeds and variants: seeds then vary task order and router randomness only (common random numbers),
and the "cache-aware cap" in `--estimate` bounds the total spend. Set it to `true` to sample the
model once per seed (that costs more).

## Dry run (SimProvider)

`SimProvider` reads the `sim:` block of each model: accuracy per task kind, role skill, plan bonus,
verifier detect and false-reject rates, context decay per 1k input tokens, and output tokens per role.
It finds the task between the `<task>` tags, decides correctness with a hash of the request, and
writes a correct or a wrong answer in the real output format. That answer then goes through the
real parser, checker and sandbox, and the cost through the real prices. These numbers are made up.
A dry run checks the pipeline only and tells you nothing about the real models.

## Limitations

- The code sandbox (`python -I`, temp dir, timeout with process-group kill, rlimits on CPU, memory,
  file size and open files) is not a security boundary: there is no network or syscall filtering.
  Run real experiments inside a container.
- Thinking tokens on current Anthropic models are billed as output. `effort: low` keeps them small.
  The estimate's output lengths (`sim.out_tokens`) are assumptions, so check them against the call
  log after a small pilot run.
- With the built-in 62-item pool, a 300-task run repeats every item about 5 times. Use `data/` for
  real conclusions.
- The server-side refusal `fallbacks` parameter is deliberately not set: a fallback model would
  silently change the arm the router is learning about. A refusal yields empty text, which counts as
  a failure, and is visible as `stop_reason` in the call log.
