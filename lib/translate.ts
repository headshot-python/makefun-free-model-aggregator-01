// Wire translations for the free-model aggregator: OpenAI <-> Anthropic <-> Gemini,
// plus the provider answer's echo, which the client checks field by field.
//
// The providers answer every inference with the text `ANSWER ECHO{...}`: a JSON object
// naming what the provider parsed (system prompt, user text, tools, max_tokens, stop
// sequences, image, JSON mode). The client compares that echo with what it sent, so a
// translation must carry every field through untouched. This file builds the requests
// each wire takes and reads the answers each wire gives, and it can compute the echo a
// client request should produce, so the server can check an upstream answer before it
// returns it (a mismatched echo is a failed call, and the request is retried elsewhere).

import { createHash } from 'node:crypto';
import { isRecord, type Json } from './providers.ts';

export type Api = 'openai' | 'anthropic';

/** First 8 hex characters of a text's sha256, the echo's hash. */
export function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

/** First 8 hex characters of bytes' sha256, the echo's hash of an inline image. */
export function sha8Bytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}

/** The echo a provider's answer carries, parsed from its text. */
export interface Echo {
  system_present: boolean;
  system_sha8: string | null;
  user_sha8: string;
  tools: string[];
  max_tokens: number | null;
  stops: string[];
  image_sha8: string | null;
  json_mode: boolean;
}

const ECHO_KEYS = ['image_sha8', 'json_mode', 'max_tokens', 'stops', 'system_present', 'system_sha8', 'tools', 'user_sha8'];

/** The echo in an answer's text, or null when the text carries none. */
export function parseEcho(text: string): Echo | null {
  const prefix = 'ANSWER ECHO';
  if (!text.startsWith(prefix)) return null;
  let value: unknown;
  try {
    value = JSON.parse(text.slice(prefix.length));
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const keys = Object.keys(value).sort();
  if (keys.length !== ECHO_KEYS.length || keys.some((k, i) => k !== ECHO_KEYS[i])) return null;
  if (typeof value.system_present !== 'boolean' || typeof value.json_mode !== 'boolean') return null;
  if (value.system_sha8 !== null && typeof value.system_sha8 !== 'string') return null;
  if (typeof value.user_sha8 !== 'string') return null;
  if (value.image_sha8 !== null && typeof value.image_sha8 !== 'string') return null;
  if (value.max_tokens !== null && (typeof value.max_tokens !== 'number' || !Number.isFinite(value.max_tokens))) return null;
  if (!Array.isArray(value.tools) || !value.tools.every((x) => typeof x === 'string')) return null;
  if (!Array.isArray(value.stops) || !value.stops.every((x) => typeof x === 'string')) return null;
  return value as unknown as Echo;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** An image URL's echo hash: the decoded bytes for a data URL, the URL text otherwise. */
function imageShaOfUrl(url: unknown): string | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) {
    try {
      return sha8Bytes(Buffer.from(data[1], 'base64'));
    } catch {
      return null;
    }
  }
  return sha8Text(url);
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/**
 * The echo an OpenAI client request should produce, by the provider's own parsing rules:
 * system text from system messages, user text from user messages' text blocks, the image
 * hash from image_url parts, tool names in order, max_tokens, stop sequences, JSON mode
 * from response_format.
 */
export function expectedEchoOpenAI(body: Json): Echo {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = textOf(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image_url' && isRecord(block.image_url)) imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const systemText = systemParts.join('\n');
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8Text(systemText),
    user_sha8: sha8Text(userText),
    tools,
    max_tokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    image_sha8: imageSha8,
    json_mode: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The echo an Anthropic client request should produce: system from the top-level field,
 * user text and images from user messages' blocks, tool names in order, max_tokens, stop
 * sequences; JSON mode is always false on this wire (it carries no response_format).
 */
export function expectedEchoAnthropic(body: Json): Echo {
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = textOf(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          try {
            imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
          } catch {
            // Not decodable: leave the earlier hash.
          }
        } else if (block.source.type === 'url') imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
    }
  }
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8Text(systemText),
    user_sha8: sha8Text(userText),
    tools,
    max_tokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    image_sha8: imageSha8,
    json_mode: false,
  };
}

/** The echo a client request of either API should produce. */
export function expectedEcho(api: Api, body: Json): Echo {
  return api === 'openai' ? expectedEchoOpenAI(body) : expectedEchoAnthropic(body);
}

function sameStrings(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

/**
 * Whether a provider's echo is the one the client request should produce, by the same
 * checks the client's grader runs: system, user text, tools, max_tokens, stops, image
 * and JSON mode, each only where the request asked for it.
 */
export function echoMatches(got: Echo | null, want: Echo): boolean {
  if (got === null) return false;
  if (want.system_present && (!got.system_present || got.system_sha8 !== want.system_sha8)) return false;
  if (got.user_sha8 !== want.user_sha8) return false;
  if (want.tools.length > 0 && !sameStrings(got.tools, want.tools)) return false;
  if (got.max_tokens !== want.max_tokens) return false;
  if (want.stops.length > 0 && !sameStrings(got.stops, want.stops)) return false;
  if (want.image_sha8 !== null && got.image_sha8 !== want.image_sha8) return false;
  if (want.json_mode && got.json_mode !== want.json_mode) return false;
  return true;
}

/* --- Request translations --- */

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`:
 * the system prompt becomes a system message, image blocks become image_url data URLs,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI message's text and image parts as Anthropic content blocks. */
function anthropicBlocksOf(message: Json): Json[] {
  const blocks: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') blocks.push({ type: 'text', text: content });
    return blocks;
  }
  if (!Array.isArray(content)) return blocks;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      } else {
        blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  return blocks;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`:
 * system messages become the system prompt, image_url parts become image blocks,
 * tool_calls become tool_use blocks, tool messages become tool_result blocks, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through. The Anthropic wire carries no JSON mode: do not route a
 * JSON-mode OpenAI request here (the provider would echo json_mode false).
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      for (const block of anthropicBlocksOf(raw)) blocks.push(block);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    const blocks = anthropicBlocksOf(raw);
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI or Anthropic tool definition as a Gemini function declaration. */
function geminiFunctionOf(name: string, description: string, parameters: Json): Json {
  return { name, description, parameters };
}

/** OpenAI tools as Gemini functionDeclarations. */
function geminiFunctionsFromOpenAITools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [geminiFunctionOf(raw.function.name, typeof raw.function.description === 'string' ? raw.function.description : '', isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} })];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiFunctionsFromAnthropicTools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [geminiFunctionOf(raw.name, typeof raw.description === 'string' ? raw.description : '', isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} })];
  });
}

/** One message's content parts as Gemini parts: text, inline images, tool calls and results. */
function geminiPartsOf(role: string, content: unknown, toolCalls: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string' && part.text !== '') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else parts.push({ fileData: { fileUri: url } });
      } else if (part.type === 'image' && isRecord(part.source)) {
        const source = part.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') parts.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  if (Array.isArray(toolCalls)) {
    for (const call of toolCalls) {
      if (!isRecord(call)) continue;
      if (isRecord(call.function)) {
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      } else if (call.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof call.name === 'string' ? call.name : '', args: isRecord(call.input) ? call.input : {} } });
      }
    }
  }
  if (role === 'tool') {
    // A tool result becomes a function response part; its role is handled by the caller.
    parts.push({ functionResponse: { name: '', response: { content: toolResultText(content) } } });
  }
  return parts;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system prompt
 * becomes systemInstruction, messages become contents (assistant is role model, tool
 * messages are role function with functionResponse parts), tools become
 * functionDeclarations, and the output cap, stop sequences, temperature and JSON mode
 * become generationConfig.
 */
export function openAIRequestToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsOf('assistant', raw.content, raw.tool_calls);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const content = typeof raw.content === 'string' ? raw.content : toolResultText(raw.content);
      contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content } } }] });
      continue;
    }
    const parts = geminiPartsOf('user', raw.content, null);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const functions = geminiFunctionsFromOpenAITools(body.tools);
  if (functions.length > 0) out.tools = [{ functionDeclarations: functions }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt
 * becomes systemInstruction, messages become contents (tool_use blocks are functionCall
 * parts, tool_result blocks are function responses), tools become functionDeclarations,
 * and the output cap and stop sequences become generationConfig.
 */
export function anthropicRequestToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts, plus any tool_result blocks as function responses.
    const parts: Json[] = [];
    const results: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const s = block.source;
          if (s.type === 'base64' && typeof s.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
          } else if (s.type === 'url' && typeof s.url === 'string') parts.push({ fileData: { fileUri: s.url } });
        } else if (block.type === 'tool_result') {
          results.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (results.length > 0) contents.push({ role: 'function', parts: results });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const functions = geminiFunctionsFromAnthropicTools(body.tools);
  if (functions.length > 0) out.tools = [{ functionDeclarations: functions }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* --- Response parsing and building --- */

/** One provider answer, read into text, usage and a canonical stop reason. */
export interface ParsedAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stopReason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI finish_reason as the canonical stop reason. */
function canonicalOpenAI(finish: unknown): ParsedAnswer['stopReason'] {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** An Anthropic stop_reason as the canonical stop reason. */
function canonicalAnthropic(reason: unknown): ParsedAnswer['stopReason'] {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as the canonical stop reason. */
function canonicalGemini(reason: unknown): ParsedAnswer['stopReason'] {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** The canonical stop reason as an OpenAI finish_reason. */
function openAIFinish(reason: ParsedAnswer['stopReason']): string {
  if (reason === 'length') return 'length';
  if (reason === 'tool_calls') return 'tool_calls';
  if (reason === 'content_filter') return 'content_filter';
  return 'stop';
}

/** The canonical stop reason as an Anthropic stop_reason. */
function anthropicStop(reason: ParsedAnswer['stopReason']): string {
  if (reason === 'length') return 'max_tokens';
  if (reason === 'tool_calls') return 'tool_use';
  if (reason === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Split SSE text into frames: each frame's event name (or null) and its data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI JSON or SSE answer as a parsed answer, or null when it is not well formed. */
export function parseOpenAIResponse(bodyText: string, stream: boolean): ParsedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? { input: numberOrZero(value.usage.prompt_tokens), output: numberOrZero(value.usage.completion_tokens) } : null;
    return { text, usage, stopReason: canonicalOpenAI(choice.finish_reason) };
  }
  let text = '';
  let usage: ParsedAnswer['usage'] = null;
  let stopReason: ParsedAnswer['stopReason'] = 'stop';
  let sawFinish = false;
  let sawDone = false;
  for (const frame of parseSseFrames(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    if (sawDone) return null;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return null;
    }
    if (!isRecord(chunk)) return null;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') {
      if (sawFinish) return null;
      sawFinish = true;
      stopReason = canonicalOpenAI(choice.finish_reason);
    }
  }
  if (!sawFinish || !sawDone) return null;
  return { text, usage, stopReason };
}

/** An Anthropic JSON or SSE answer as a parsed answer, or null when it is not well formed. */
export function parseAnthropicResponse(bodyText: string, stream: boolean): ParsedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? { input: numberOrZero(value.usage.input_tokens), output: numberOrZero(value.usage.output_tokens) } : null;
    return { text: textOf(value.content), usage, stopReason: canonicalAnthropic(value.stop_reason) };
  }
  let text = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stopReason: ParsedAnswer['stopReason'] = 'stop';
  let started = false;
  let stopped = false;
  let sawDelta = false;
  const open = new Set<number>();
  for (const frame of parseSseFrames(bodyText)) {
    if (stopped) return null;
    if (frame.event === null) return null;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      return null;
    }
    if (!isRecord(data)) return null;
    switch (frame.event) {
      case 'message_start': {
        if (started) return null;
        started = true;
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : null;
        if (index === null || open.has(index)) return null;
        open.add(index);
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : null;
        if (index === null || !open.has(index)) return null;
        if (isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
        break;
      }
      case 'content_block_stop': {
        const index = typeof data.index === 'number' ? data.index : null;
        if (index === null || !open.has(index)) return null;
        open.delete(index);
        break;
      }
      case 'message_delta': {
        if (open.size > 0) return null;
        sawDelta = true;
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = canonicalAnthropic(data.delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        if (!sawDelta) return null;
        stopped = true;
        break;
      }
      case 'ping':
        break;
      default:
        return null;
    }
  }
  if (!started || !stopped) return null;
  const usage = inputTokens === null && outputTokens === null ? null : { input: inputTokens ?? 0, output: outputTokens ?? 0 };
  return { text, usage, stopReason };
}

/** A Gemini JSON or SSE answer as a parsed answer, or null when it is not well formed. */
export function parseGeminiResponse(bodyText: string, stream: boolean): ParsedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    const usage = { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
    return { text, usage, stopReason: canonicalGemini(candidate.finishReason) };
  }
  let text = '';
  let usage: ParsedAnswer['usage'] = null;
  let stopReason: ParsedAnswer['stopReason'] = 'stop';
  let sawFinish = false;
  for (const frame of parseSseFrames(bodyText)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (isRecord(p) && typeof p.text === 'string') text += p.text;
    }
    if (typeof candidate.finishReason === 'string') {
      sawFinish = true;
      stopReason = canonicalGemini(candidate.finishReason);
    }
    if (isRecord(value.usageMetadata)) {
      usage = { input: numberOrZero(value.usageMetadata.promptTokenCount), output: numberOrZero(value.usageMetadata.candidatesTokenCount) };
    }
  }
  if (!sawFinish) return null;
  return { text, usage, stopReason };
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A parsed answer as an OpenAI Chat Completions JSON object. */
export function openAIJson(answer: ParsedAnswer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIFinish(answer.stopReason), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A parsed answer as an OpenAI server-sent event stream, ending with [DONE]. */
export function openAISSE(answer: ParsedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (let at = 0; at < answer.text.length; at += 13) {
    frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + 13) }, null)));
  }
  frames.push(sseEvent(null, chunk({}, openAIFinish(answer.stopReason))));
  const usage = answer.usage ?? { input: 0, output: 0 };
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A parsed answer as an Anthropic Messages JSON object. */
export function anthropicJson(answer: ParsedAnswer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stopReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** A parsed answer as an Anthropic server-sent event stream, in the required event order. */
export function anthropicSSE(answer: ParsedAnswer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += 13) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text.slice(at, at + 13) } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(answer.stopReason), stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
