// Subprocess integration tests for the headless `opencode run` BACKGROUND-SUBAGENT DRAIN.
//
// Headless `run` used to break on the first session idle and exit, orphaning any background `task`
// subagents (they run in a detached scope, so they never defer idle). The drain keeps the run alive
// while any launched background task is still pending, so its completion re-prompt can drive a
// dependent follow-up (e.g. a discovery "report" step after the per-source collectors finish).
//
// These tests spawn the real CLI against the in-process TestLLMServer, routing each turn by request
// content (main vs the worker sub-session vs the completion re-prompt).
import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { reply } from "../../lib/llm-server"
import { cliIt } from "../../lib/cli-process"
import { testProviderConfig } from "../../lib/test-provider"

const WORKER_MARKER = "WORKER_TASK_MARKER_9f2a"

// Config with a `worker` subagent the top-level agent can spawn, + the background-subagents experiment on.
function bgEnv(llmUrl: string): Record<string, string> {
  return {
    OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS: "true",
    OPENCODE_CONFIG_CONTENT: JSON.stringify({
      ...testProviderConfig(llmUrl),
      agent: { worker: { mode: "subagent" } },
    }),
  }
}

// Match on the accumulated request messages (system + conversation).
const msgs = (hit: { body: Record<string, unknown> }) => JSON.stringify(hit.body.messages ?? [])

describe("opencode run — background-subagent drain", () => {
  // THE feature test: a top-level agent launches a background `task`, then AFTER it completes emits a
  // dependent follow-up. Without the drain the run exits at the first idle and the follow-up never runs
  // (so its text never reaches stdout). With the drain, the background completion re-prompts the agent
  // and the follow-up IS produced.
  cliIt.concurrent(
    "stays alive for a background subagent and runs the dependent follow-up",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        // main's FIRST turn -> launch the worker in the background
        yield* llm.pushMatch(
          (h) => msgs(h).includes("run the worker") && !msgs(h).includes("working in the background") && !msgs(h).includes(WORKER_MARKER),
          reply()
            .tool("task", {
              description: "collect",
              subagent_type: "worker",
              prompt: `${WORKER_MARKER} do the work`,
              background: true,
            })
            .item(),
        )
        // main's continuation right after launch (tool result present, no completion yet) -> end the turn
        yield* llm.pushMatch(
          (h) => msgs(h).includes("working in the background") && !msgs(h).includes("completed"),
          reply().text("launched the worker in the background").stop().item(),
        )
        // the worker sub-session -> produce its result and finish
        yield* llm.pushMatch(
          (h) => msgs(h).includes(WORKER_MARKER),
          reply().text("worker finished collecting").stop().item(),
        )
        // main RE-PROMPT after the worker completes -> the dependent follow-up (only reachable via the drain)
        yield* llm.pushMatch(
          (h) => msgs(h).includes("Background task completed"),
          reply().text("FOLLOWUP_AFTER_WORKER done").stop().item(),
        )

        const result = yield* opencode.run("run the worker", {
          extraArgs: ["--dangerously-skip-permissions"],
          env: bgEnv(llm.url),
          timeoutMs: 60_000,
        })
        opencode.expectExit(result, 0)
        // The follow-up text only exists if the run stayed alive through the background completion.
        expect(result.stdout).toContain("FOLLOWUP_AFTER_WORKER done")
      }),
    90_000,
  )

  // Blast-radius guard: with the flag ON but NO background task launched (the common case — betel sets
  // the flag globally), the run must still exit on idle exactly as before. bgLaunched stays empty, so
  // the drain is a no-op.
  cliIt.concurrent(
    "flag on but no background task launched: exits on idle as before",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.text("plain answer")
        const result = yield* opencode.run("say hi", { env: bgEnv(llm.url) })
        opencode.expectExit(result, 0)
        expect(result.stdout).toBe("plain answer\n")
      }),
    60_000,
  )

  // Same as above but a FOREGROUND task (background omitted): a foreground task blocks the launcher, so
  // it never reaches idle while pending; the run exits once the (single) turn completes — unchanged.
  cliIt.concurrent(
    "foreground task is unaffected by the drain",
    ({ llm, opencode }) =>
      Effect.gen(function* () {
        yield* llm.pushMatch(
          (h) => msgs(h).includes("run the worker") && !msgs(h).includes(WORKER_MARKER),
          reply().tool("task", { description: "collect", subagent_type: "worker", prompt: `${WORKER_MARKER} do it` }).item(),
        )
        yield* llm.pushMatch((h) => msgs(h).includes(WORKER_MARKER), reply().text("worker done fg").stop().item())
        yield* llm.pushMatch(
          (h) => msgs(h).includes("</task>") || msgs(h).includes("task_result"),
          reply().text("MAIN_AFTER_FG done").stop().item(),
        )
        const result = yield* opencode.run("run the worker", {
          extraArgs: ["--dangerously-skip-permissions"],
          env: bgEnv(llm.url),
          timeoutMs: 60_000,
        })
        opencode.expectExit(result, 0)
        expect(result.stdout).toContain("MAIN_AFTER_FG done")
      }),
    90_000,
  )
})
