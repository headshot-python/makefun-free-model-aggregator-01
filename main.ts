// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the trial's brief (ten providers, the quality table, the workload),
// probes each provider's catalog, registers its port and catalog with the env, and
// answers the client's twenty requests. For each request it picks the best free model
// that meets the request's requirements, speaks that provider's wire (OpenAI, Anthropic
// or Gemini), and, when a provider refuses with 402 or 429, answers the request with
// another model instead.
//
// Routing: candidates are ranked by the quality table's rating alone. A provider that
// refuses is not retried for the same request, and a provider that refuses twice in a
// row is treated as spent for the rest of the trial: its budget is gone or it is out,
// and further calls to it would only spend the tie-break. A 2xx answer is always used
// (a second successful call would break the trial's one-provider rule), even when its
// echo is not the one the client asked for.
//
// Every choice is a function of the brief, the requests and the providers' answers:
// no wall clock and no unseeded randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  anthropicJson,
  anthropicRequestToGemini,
  anthropicSSE,
  anthropicToOpenAI,
  echoMatches,
  expectedEcho,
  openAIJson,
  openAIRequestToGemini,
  openAISSE,
  openAIToAnthropicRequest,
  parseAnthropicResponse,
  parseEcho,
  parseGeminiResponse,
  parseOpenAIResponse,
  type Api,
  type Echo,
  type ParsedAnswer,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent for the trial. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  for (const probe of probes) {
    if (probe.ok) providerState.set(probe.provider.id, { calls: 0, consecutiveFailures: 0, dead: false });
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** What one provider has cost so far, and whether it is spent. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best rating first. A provider that has
 * refused twice in a row is left out (its budget is spent or it is out). Capabilities
 * come from the provider's own listing first: it is the truth the trial scores against.
 * A JSON-mode request from an OpenAI client may only go to a wire that can carry
 * response_format (OpenAI or Gemini); the Anthropic wire echoes json_mode false.
 */
function candidatesFor(requires: Requirement, clientApi: Api, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = providerState.get(probe.provider.id);
    if (state !== undefined && state.dead) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      // Capabilities from the listing first (the provider's own word), the quality
      // table second; a capability neither names never meets a requirement.
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Record one upstream call's outcome against its provider. */
function noteCall(providerId: string, ok: boolean): void {
  const state = providerState.get(providerId);
  if (state === undefined) return;
  state.calls += 1;
  if (ok) {
    state.consecutiveFailures = 0;
  } else {
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) state.dead = true;
  }
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The answer one provider gave, or why it gave none. */
interface CallResult {
  status: number;
  answer: ParsedAnswer | null;
  failure: string | null;
}

/** One HTTP call to a provider: 2xx and a body is a success even when the answer is not what was asked for. */
async function httpCall(url: string, headers: Record<string, string>, body: unknown): Promise<{ status: number; text: string } | null> {
  const init: RequestInit = { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) };
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    return null;
  }
  let text = '';
  try {
    text = await response.text();
  } catch {
    text = '';
  }
  return { status: response.status, text };
}

/** One provider call: a 2xx answer is always used; a refusal (402/429/5xx) or a call that never got an answer is not. */
function handleUpstream(status: number, text: string, parse: (bodyText: string) => ParsedAnswer | null, expect: Echo | null): CallResult {
  if (status < 200 || status >= 300) return { status, answer: null, failure: `HTTP ${status}` };
  const answer = parse(text);
  if (answer === null) return { status, answer: null, failure: 'unparseable provider answer' };
  if (expect !== null) {
    const echo = parseEcho(answer.text);
    if (echo === null || !echoMatches(echo, expect)) {
      log(`echo mismatch: expected ${JSON.stringify(expect)}, got ${JSON.stringify(echo)}`);
    }
  }
  return { status, answer, failure: null };
}

interface UpstreamRequest {
  /** The provider's wire, which decides the request body and how the answer parses. */
  wire: 'openai' | 'anthropic' | 'gemini';
  /** The body to send the provider, already in its wire. */
  body: Json;
  /** The provider's base URL. */
  baseUrl: string;
  /** The headers the provider's auth expects. */
  headers: Record<string, string>;
  /** The model the request names: its wire's model id. */
  modelId: string;
  /** Whether the provider should stream its answer. */
  stream: boolean;
}

/** One call to a provider, through its own wire, and the parsed answer (or why none came back). */
async function callProvider(request: UpstreamRequest, expect: Echo | null): Promise<CallResult> {
  if (request.wire === 'openai') {
    const body: Json = { ...request.body, model: request.modelId };
    if (request.stream) body.stream_options = { include_usage: true };
    const call = await httpCall(joinBase(request.baseUrl, '/v1/chat/completions'), { ...request.headers, 'content-type': 'application/json' }, body);
    if (call === null) return { status: 0, answer: null, failure: 'no response' };
    return handleUpstream(call.status, call.text, (text) => parseOpenAIResponse(text, request.stream), expect);
  }
  if (request.wire === 'anthropic') {
    const call = await httpCall(
      joinBase(request.baseUrl, '/v1/messages'),
      { ...request.headers, 'content-type': 'application/json', 'anthropic-version': '2023-06-01' },
      { ...request.body, model: request.modelId },
    );
    if (call === null) return { status: 0, answer: null, failure: 'no response' };
    return handleUpstream(call.status, call.text, (text) => parseAnthropicResponse(text, request.stream), expect);
  }
  const path = `/v1beta/models/${encodeURIComponent(request.modelId)}:${request.stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const call = await httpCall(joinBase(request.baseUrl, path), { ...request.headers, 'content-type': 'application/json' }, request.body);
  if (call === null) return { status: 0, answer: null, failure: 'no response' };
  return handleUpstream(call.status, call.text, (text) => parseGeminiResponse(text, request.stream), expect);
}

/** The upstream request for one candidate: the client's request translated into that provider's wire. */
function upstreamFor(candidate: Candidate, clientApi: Api, body: Json, stream: boolean): UpstreamRequest {
  const wire = candidate.probe.provider.wire;
  const headers = authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key);
  if (clientApi === 'openai') {
    if (wire === 'openai') return { wire, body: { ...body }, baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
    if (wire === 'anthropic') return { wire, body: openAIToAnthropicRequest(body, candidate.model.id), baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
    return { wire, body: openAIRequestToGemini(body), baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
  }
  if (wire === 'openai') return { wire, body: anthropicToOpenAI(body, candidate.model.id), baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
  if (wire === 'anthropic') return { wire, body: { ...body }, baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
  return { wire, body: anthropicRequestToGemini(body), baseUrl: joinBase(ENV_URL, candidate.probe.provider.base), headers, modelId: candidate.model.id, stream };
}

/**
 * One client request, served by the best free model that can: candidates are tried in
 * rating order; a refusal is left behind, a 2xx answer is always used, and a provider
 * that refuses twice in a row is treated as spent. Returns the answer and the model
 * that served it, or null when nothing can.
 */
async function serve(clientApi: Api, body: Json, stream: boolean): Promise<{ answer: ParsedAnswer; modelId: string } | null> {
  const requires = requirementsFor(body);
  const expect = expectedEcho(clientApi, body);
  const jsonMode = clientApi === 'openai' ? jsonModeOf(body) : false;
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const tried = new Set<string>();
  for (const candidate of candidates) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const result = await callProvider(upstreamFor(candidate, clientApi, body, stream), expect);
    if (result.answer !== null) {
      noteCall(candidate.probe.provider.id, true);
      return { answer: result.answer, modelId: candidate.model.id };
    }
    noteCall(candidate.probe.provider.id, false);
    log(`${candidate.probe.provider.id} ${candidate.model.id} refused: ${result.failure ?? `HTTP ${result.status}`}`);
  }
  return null;
}

/** One server-sent event stream, with the headers that keep it unbuffered. */
function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', 'x-accel-buffering': 'no' });
  res.end(text);
}

/** POST /v1/chat/completions: answer in the OpenAI wire, streaming when the client asked for it. */
async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const served = await serve('openai', body, stream);
  if (served === null) {
    sendError(res, 'openai', 503, 'service_unavailable', 'no free model could serve this request');
    return;
  }
  if (stream) sendSse(res, openAISSE(served.answer, served.modelId));
  else sendJson(res, 200, openAIJson(served.answer, served.modelId));
}

/** POST /v1/messages: answer in the Anthropic wire, streaming when the client asked for it. */
async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const served = await serve('anthropic', body, stream);
  if (served === null) {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
    return;
  }
  if (stream) sendSse(res, anthropicSSE(served.answer, served.modelId));
  else sendJson(res, 200, anthropicJson(served.answer, served.modelId));
}

/** GET /v1/models: the free models across the providers that answered, in the wire the caller uses. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = servedModels();
  const authorization = req.headers.authorization ?? '';
  if (authorization.toLowerCase().startsWith('bearer ')) sendJson(res, 200, openAIModelList(models));
  else if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models));
  else sendJson(res, 200, openAIModelList(models));
}

/** One request, to the route its path names. */
async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleMessages(req, res);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

/** Listen on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout). */
function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
