import { describe, expect, test } from "bun:test"
import { scanBuffer, type ScanPart } from "./groq-toolcall"

const KNOWN = new Set(["run_actions", "discover_actions", "read"])

// Feed chunks through scanBuffer exactly as the middleware does (running buffer + rest), then flush.
function runStream(chunks: string[], known: ReadonlySet<string> = KNOWN) {
  let buf = ""
  let text = ""
  const tools: Array<{ name: string; args: unknown }> = []
  for (const c of chunks) {
    buf += c
    const { parts, rest } = scanBuffer(buf, known)
    buf = rest
    for (const p of parts) p.type === "text" ? (text += p.text) : tools.push({ name: p.name, args: p.args })
  }
  text += buf // text-end / stream flush emits any held remainder as text
  return { text, tools }
}

const one = (s: string, known = KNOWN): ScanPart[] => scanBuffer(s, known).parts

describe("scanBuffer — recognised calls", () => {
  test("canonical <function=NAME>{json}</function>", () => {
    expect(one(`<function=discover_actions>{"use_case":"read email"}</function>`)).toEqual([
      { type: "tool", name: "discover_actions", args: { use_case: "read email" } },
    ])
  })
  test("mangled 8B <function>NAME>{}</function>", () => {
    expect(one(`<function>run_actions>{}</function>`)).toEqual([{ type: "tool", name: "run_actions", args: {} }])
  })
  test("python_tag name+parameters", () => {
    expect(one(`<|python_tag|>{"name":"run_actions","parameters":{"actions":[]}}`)).toEqual([
      { type: "tool", name: "run_actions", args: { actions: [] } },
    ])
  })
  test("nested + string-brace args balance correctly", () => {
    const r = one(`<function=run_actions>{"q":"a{b}c \\" }","x":[{"y":1}]}</function>`)
    expect(r).toEqual([{ type: "tool", name: "run_actions", args: { q: 'a{b}c " }', x: [{ y: 1 }] } }])
  })
})

describe("scanBuffer — review fixes", () => {
  test("[2] prose BEFORE the marker is emitted, not dropped", () => {
    expect(one(`Reading your inbox. <function=run_actions>{}</function>`)).toEqual([
      { type: "text", text: "Reading your inbox. " },
      { type: "tool", name: "run_actions", args: {} },
    ])
  })
  test("[3] unknown-name marker becomes text and a LATER real call is still found", () => {
    const r = one(`<function=not_a_tool>{} then <function=run_actions>{"a":1}</function>`)
    expect(r.filter((p) => p.type === "tool")).toEqual([{ type: "tool", name: "run_actions", args: { a: 1 } }])
    // the bogus marker + its json survive as text (nothing executed)
    expect(r.some((p) => p.type === "text" && p.text.includes("not_a_tool"))).toBe(true)
  })
  test("[1] ```json / bare {\"name\"} example is NOT executed (stays text)", () => {
    expect(one('Example: ```json {"name":"read","parameters":{"path":"/etc"}}').every((p) => p.type === "text")).toBe(
      true,
    )
    expect(one(`Call {"name":"read","parameters":{}} to read.`).every((p) => p.type === "text")).toBe(true)
  })
  test("[5] prose containing '<function' words / '{\"name\"' does NOT latch — text flushes", () => {
    const { text, tools } = runStream(["The read ", "function loads a file. ", "Done."])
    expect(text).toBe("The read function loads a file. Done.")
    expect(tools).toEqual([])
  })
  test("[4] bare (unprefixed) name resolves to the unique prefixed tool", () => {
    const known = new Set(["apps_discover_actions", "apps_run_actions"])
    expect(one(`<function=discover_actions>{}</function>`, known)).toEqual([
      { type: "tool", name: "apps_discover_actions", args: {} },
    ])
  })
  test("[4] ambiguous bare name (two matches) is NOT converted", () => {
    const known = new Set(["a_run", "b_run"])
    expect(one(`<function=run>{}</function>`, known).every((p) => p.type === "text")).toBe(true)
  })
})

describe("scanBuffer — streaming (split across deltas, no loss / no dupe)", () => {
  test("marker split mid-name is held then recovered; surrounding prose preserved", () => {
    const { text, tools } = runStream(["Reading now. <function=run_", 'actions>{"actions":', "[]}</function> done"])
    expect(text).toBe("Reading now.  done")
    expect(tools).toEqual([{ name: "run_actions", args: { actions: [] } }])
  })
  test("incomplete JSON across deltas doesn't emit a broken call", () => {
    const { text, tools } = runStream([`<function=discover_actions>{"use_case":"re`, `ad my email"}</function>`])
    expect(tools).toEqual([{ name: "discover_actions", args: { use_case: "read my email" } }])
    expect(text).toBe("")
  })
  test("trailing partial marker at stream end flushes as text (never dropped)", () => {
    expect(runStream(["hello <functio"]).text).toBe("hello <functio")
  })
  test("two calls in one buffer both recovered", () => {
    const { tools } = runStream([`<function=discover_actions>{}</function><function=run_actions>{"a":1}</function>`])
    expect(tools).toEqual([
      { name: "discover_actions", args: {} },
      { name: "run_actions", args: { a: 1 } },
    ])
  })
  test("marker with no JSON body is emitted as text, scanning continues", () => {
    const { text, tools } = runStream(["<function=run_actions> not a call, just prose."])
    expect(tools).toEqual([])
    expect(text).toBe("<function=run_actions> not a call, just prose.")
  })
  test("plain prose passes straight through", () => {
    expect(runStream(["Here are your 3 most recent emails: ...done."])).toEqual({
      text: "Here are your 3 most recent emails: ...done.",
      tools: [],
    })
  })
})
