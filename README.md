# makefun-free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier providers. For each of the client's twenty requests it picks the best free model that meets the request's requirements (tools, vision, JSON mode, long context), speaks that provider's own wire (OpenAI, Anthropic or Gemini), and fails over to the next best model when a provider refuses with 402 or 429.

Run it with `node main.ts` (it registers its catalog with the env and exits when the workload is done); `node main.ts --list` prints the catalog it found.
