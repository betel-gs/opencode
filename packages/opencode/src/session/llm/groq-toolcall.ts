import type { LanguageModelV3Middleware, LanguageModelV3StreamPart } from "@ai-sdk/provider"

/**
 * Recover tool calls that a Groq-served Llama/Qwen model emitted as ASSISTANT TEXT instead of a
 * structured `tool_calls` entry. Groq's server-side tool parser reliably structures calls for
 * gpt-oss but frequently misses them for the llama/qwen chat-template variants, so the invocation
 * arrives as content text. A gated AI SDK `wrapStream` middleware (groqTextToolCallMiddleware)
 * scans the streaming text and converts a recognised call into a real `tool-call` stream part.
 *
 * Only the DISTINCTIVE Meta/Llama tool-call markers are treated as invocations — a bare `{...}` or a
 * ```json fence is NOT, because models routinely show those as documentation/examples and executing
 * an example would be a real side-effect. Markers handled:
 *   <function=NAME>{json}</function>      canonical custom-tool
 *   <function>NAME>{json}</function>      mangled 8B (missing '=', stray '>')
 *   <|python_tag|>{"name":"NAME","parameters"|"arguments":{...}}   llama-3.1 built-in/code path
 */

// Named markers carry the tool name in the regex; the python_tag marker carries it inside the JSON.
const OPENERS: Array<{ re: RegExp; named: boolean }> = [
  { re: /<function\s*=\s*([A-Za-z0-9_.-]+)\s*>/, named: true },
  { re: /<function\s*>\s*([A-Za-z0-9_.-]+)\s*>/, named: true }, // mangled 8B
  { re: /<\|python_tag\|>/, named: false },
]

type ScanJson = { parsed: unknown; end: number } | "incomplete" | "notjson"

/** From `start` (skipping whitespace) parse one balanced JSON value. Distinguishes a value still
 * streaming ("incomplete") from text that will never be JSON ("notjson"). */
function scanJson(buf: string, start: number): ScanJson {
  let i = start
  while (i < buf.length && /\s/.test(buf[i]!)) i++
  if (i >= buf.length) return "incomplete" // only whitespace so far — args may still arrive
  if (buf[i] !== "{" && buf[i] !== "[") return "notjson"
  let depth = 0
  let inStr = false
  let esc = false
  for (let j = i; j < buf.length; j++) {
    const ch = buf[j]!
    if (inStr) {
      if (esc) esc = false
      else if (ch === "\\") esc = true
      else if (ch === '"') inStr = false
      continue
    }
    if (ch === '"') inStr = true
    else if (ch === "{" || ch === "[") depth++
    else if (ch === "}" || ch === "]") {
      depth--
      if (depth === 0) {
        try {
          return { parsed: JSON.parse(buf.slice(i, j + 1)), end: j + 1 }
        } catch {
          return "notjson"
        }
      }
    }
  }
  return "incomplete" // braces opened but not yet balanced
}

/** Resolve a possibly-BARE tool name (the Groq bridge lets the model drop the `<server>_` prefix)
 * to an actual offered tool: exact match, else the UNIQUE offered tool whose name ends `_<bare>`. */
function resolveName(name: string, known: ReadonlySet<string>): string | null {
  if (known.has(name)) return name
  const hits = [...known].filter((k) => k.endsWith("_" + name))
  return hits.length === 1 ? hits[0]! : null
}

function firstOpener(buf: string, from: number) {
  let best: { at: number; opener: (typeof OPENERS)[number]; m: RegExpMatchArray } | null = null
  for (const opener of OPENERS) {
    const m = buf.slice(from).match(opener.re)
    if (m && m.index !== undefined) {
      const at = from + m.index
      if (best === null || at < best.at) best = { at, opener, m }
    }
  }
  return best
}

/** The index (>= from) at which the tail might still grow into a marker and so must be HELD; buf.length
 * if nothing at the tail could — everything before it is safe to emit as text. Handles a marker split
 * mid-token across deltas (`<functio`, `<function=run_`, `<|python_ta`) without latching on stray `<`. */
function heldFrom(buf: string, from: number): number {
  const lt = buf.lastIndexOf("<")
  if (lt < from) return buf.length // no `<` in the unconsumed tail -> nothing to hold
  const tail = buf.slice(lt)
  if ("<|python_tag|>".startsWith(tail)) return lt // partial python_tag
  if (/^<function(?:\s*[=>]?\s*[\w.-]*)?$/.test(tail)) return lt // "<function" + optional =/>/name, no close
  if (/^<f?u?n?c?t?i?o?n?$/.test(tail)) return lt // partial "<function" letters
  return buf.length
}

export type ScanPart = { type: "text"; text: string } | { type: "tool"; name: string; args: unknown }

/**
 * Split an accumulating text buffer into ordered text/tool parts plus the unconsumed `rest` to keep
 * buffering. Emits prose BEFORE a marker, converts a complete recognised call, treats an
 * unknown-name or non-JSON marker as literal text (so a real call later in the buffer is still
 * found), and holds back only a genuinely-incomplete call or a trailing partial marker.
 */
export function scanBuffer(buf: string, known: ReadonlySet<string>): { parts: ScanPart[]; rest: string } {
  const parts: ScanPart[] = []
  let i = 0
  while (i < buf.length) {
    const found = firstOpener(buf, i)
    if (!found) {
      const cut = heldFrom(buf, i)
      if (cut > i) parts.push({ type: "text", text: buf.slice(i, cut) })
      return { parts, rest: buf.slice(cut) }
    }
    if (found.at > i) parts.push({ type: "text", text: buf.slice(i, found.at) })
    const afterOpener = found.at + found.m[0].length
    const scanned = scanJson(buf, afterOpener)
    if (scanned === "incomplete") return { parts, rest: buf.slice(found.at) } // wait for the args
    if (scanned === "notjson") {
      // Marker with no JSON body -> not a call; emit the marker as text and continue scanning.
      parts.push({ type: "text", text: buf.slice(found.at, afterOpener) })
      i = afterOpener
      continue
    }
    let end = scanned.end
    const tail = buf.slice(scanned.end).match(/^\s*<\/function>/)
    if (tail) end = scanned.end + tail[0].length
    let name: string | undefined
    let args: unknown
    if (found.opener.named) {
      name = found.m[1]
      args = scanned.parsed
    } else {
      const obj = scanned.parsed as Record<string, unknown>
      if (obj && typeof obj === "object" && typeof obj.name === "string") {
        name = obj.name
        args = "parameters" in obj ? obj.parameters : "arguments" in obj ? obj.arguments : {}
      }
    }
    const resolved = name ? resolveName(name, known) : null
    if (resolved) parts.push({ type: "tool", name: resolved, args: args ?? {} })
    else parts.push({ type: "text", text: buf.slice(found.at, end) }) // unknown tool -> literal text
    i = end
  }
  return { parts, rest: "" }
}

/**
 * Gated AI SDK middleware (attach only for Groq llama/qwen) that rewrites text-emitted tool calls
 * into real `tool-call` stream parts. Well-behaved models never reach this; ordinary prose flows
 * straight through, and held text is always flushed within its text block (before text-end) so no
 * content is dropped and no delta is emitted after its block closed.
 */
export function groqTextToolCallMiddleware(known: ReadonlySet<string>): LanguageModelV3Middleware {
  return {
    specificationVersion: "v3" as const,
    async wrapStream({ doStream }) {
      const { stream, ...rest } = await doStream()
      let buf = ""
      let n = 0
      let textId: string | undefined
      const transform = new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
        transform(part, controller) {
          if (part.type === "text-start") {
            textId = part.id
            controller.enqueue(part)
            return
          }
          if (part.type === "text-end") {
            if (buf) controller.enqueue({ type: "text-delta", id: part.id, delta: buf }) // flush IN block
            buf = ""
            controller.enqueue(part)
            return
          }
          if (part.type !== "text-delta") {
            controller.enqueue(part)
            return
          }
          textId = part.id
          buf += part.delta
          const { parts, rest } = scanBuffer(buf, known)
          buf = rest
          for (const p of parts) {
            if (p.type === "text") {
              if (p.text) controller.enqueue({ type: "text-delta", id: part.id, delta: p.text })
            } else {
              controller.enqueue({
                type: "tool-call",
                toolCallId: `llama_tc_${n++}`,
                toolName: p.name,
                input: JSON.stringify(p.args ?? {}),
              })
            }
          }
        },
        flush(controller) {
          // Only reached if the stream ended without a text-end; never drop buffered content.
          if (buf && textId !== undefined) controller.enqueue({ type: "text-delta", id: textId, delta: buf })
        },
      })
      return { stream: stream.pipeThrough(transform), ...rest }
    },
  }
}
