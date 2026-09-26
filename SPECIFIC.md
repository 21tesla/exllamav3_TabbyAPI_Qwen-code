# SPECIFIC.md — the 2026-09-26 `ndlite` probe

One observed session, recorded in full because it is the largest body of evidence on what this
checkpoint actually does wrong when it emits tool calls. **Everything below is measured from that
session**, not inferred: a code audit of `~/software/ndlite` run on
`DeepSeek-V4-Flash-0731-exl3-2.32bpw` through the proxy, with the context deliberately pushed large
while `tabby_watch.py` tailed both the transcript and the proxy journal.

Session: `7edd9e8c-aa41-47a4-9bfb-479a354fb2fd`
Transcript: `~/.qwen/projects/-home-logan-software-ndlite/chats/7edd9e8c-*.jsonl`
Window: `2026-09-26T20:30:13Z` → `23:06:40Z` (2 h 36 min, 418 records)
Context reached: ~115 000 tokens (11.5 % of 1 M) — user-supplied, see §10

The session ran in **two phases with a 132-minute break** between them: the audit proper
(`20:30`–`20:49Z`), then a documentation phase (`23:01`–`23:06Z`) after the model was resumed. The
break matters — §11 shows it is where the width correlation breaks down.

## 1. Headline

| | |
|---|---|
| Assistant turns carrying calls | 33 |
| Tool calls emitted | 150 |
| Calls naming a tool that does not exist | **9 = 6.0 %** |
| Calls whose client-visible text leaked `<tool_call>` markup | **0** |
| Upstream non-2xx responses | **0** (51 × `HTTP/1.1 200`) |
| Proxy `[WARNING]` lines | 9 — all of them the undeclared-name check |
| Journal hits for `repetition` / `unparsed` / `malformed` | **0** |
| `api_error` records in the transcript | **0** |
| Turns aborted by the client | **0** |

Every one of the nine bad calls was absorbed by the client as a recoverable `tool_not_registered`
result and the turn continued. The failure that motivated §2.11 — a payload with no parseable call
at all — did **not** recur, across 51 upstream completions. The audit itself succeeded: three
commits, pushed (§12).

## 2. Method, so it can be repeated

```bash
# start the observer (it is a resident monitor; wind it down deliberately)
~/software/exllamav3-anemone/venv/bin/python \
  ~/software/exllamav3_TabbyAPI_Qwen-code/tabby_watch.py

# the proxy's side, which transcripts do not record
journalctl --user -u tabby-proxy@home-logan.service --since "16:30" --until "19:15" -o cat
```

Transcript timestamps are **UTC**; `journalctl --since` is **local**. Convert, or the query returns
nothing and looks like a clean run. This was hit twice before it was noticed.

The per-turn shape comes from the transcript directly — each `functionCall` part is
`{"id", "name", "args"}`, and one assistant record can carry many of them:

```python
for r in recs:
    if r.get("type") != "assistant": continue
    cs = [p["functionCall"] for p in r["message"]["parts"] if p.get("functionCall")]
```

Two traps when doing this by hand: the field is `args`, **not** `arguments` (projecting the wrong
one yields `null` for every call and makes a populated turn look empty), and a loose digit grep over
`journalctl -o cat` matches the millisecond field of the proxy's own timestamp — `16:37:22,454`
looks like a `454`. Anchor a status match to `HTTP/1\.1 (\d{3})`.

A third trap, hit while checking this session's own notes: a substring count is not a count of a
thing. `grep -c PDATE.md` returns 80 when the text holds 79 `UPDATE.md` and **one** genuinely
truncated `PDATE.md`, because `UPDATE.md` contains `PDATE.md`. Use a negative lookbehind
(`(?<!U)PDATE\.md`) when the question is whether a leading character was dropped.

## 3. The nine inventions

| time (Z) | phase | batch width | position | name | args |
|---|---|---|---|---|---|
| 20:32:36 | audit | 9 | 0 | `tool_result` | `{}` |
| 20:36:12 | audit | 8 | 0 | `on` | `{}` |
| 20:41:30 | audit | 2 | 0 | `arguments` | `{}` |
| 20:41:30 | audit | 2 | 1 | `name` | `{}` |
| 20:44:06 | audit | 28 | 0 | `call` | `{}` |
| 20:45:32 | audit | 13 | 0 | `ead` | `{}` |
| 20:45:32 | audit | 13 | 1 | `editing` | `{}` |
| 20:48:25 | audit | 9 | 0 | `runtool_call` | `{}` |
| **23:03:13** | **docs** | **2** | **0** | **`call`** | `{}` |

All nine carried empty arguments and all nine drew the same client error, verbatim:

```
Tool "X" not found in registry. Tools must use the exact names that are registered.
Did you mean one of: "…"?
```

Proxy side, for each of them, the same two lines in this order:

```
[WARNING] Tool call 'X' is not one of the 20 tools this request declared (batch=9); the client cannot dispatch it
[INFO]    Intercepted and parsed N tool call(s) from content: ['X', 'read_file', …]
```

That ordering is the diagnostic: the payload **parsed cleanly onto a name the client rejected**.
The separator between this and a parser fault is that the invented string is present in the
`Intercepted and parsed` list — it entered the extractor as that name rather than being assembled
from fragments.

`call` is the **only name to recur** (once per phase), and it did so after 2 h 15 min of wall time.
Repetition across a phase boundary, at the same position, is the strongest evidence in this session
that the stub is a stable decoding attractor rather than a random slip.

## 4. Three families

The names are not random. They fall into three groups, which is the detail §2.8's phrase
"one-off names" does not capture:

**Wire-format vocabulary** — `tool_result`, `arguments`, `name`, `call` ×2. These are the field
names and content-block types of the OpenAI/Anthropic tool-calling formats. `tool_result` is the
Anthropic content-block type for tool output; `arguments`/`name` are the keys of a call object;
`call_` is the client's own id prefix. The model is emitting the *protocol's own words* in the name
slot.

**Prose and character debris** — `on`, `ead`, `editing`. Fragments of nearby English
("...is **on** the...", "Now I need to **edit**ing"), and in `ead` a truncation missing its first
character. `ead` is the only one that could have been a slicing artefact, and §5 rules that out.

**Fused keyword pair** — `runtool_call`, once. `run` (from `run_shell_command`) and `tool_call`
(the grammar's own tag) collapsed into one name. Both halves are vocabulary that was in flight in
that batch, and the model merged them rather than choosing.

**Predicted, not observed.** §4 previously predicted `type`, `id`, `function`, `parameters` and a
bare `run` would appear if the wire-vocabulary family was real. **None of them appeared**, in this
session or anywhere in `~/.qwen/projects`. The prediction is not falsified — 9 samples across five
narrow families is not enough to expect any particular member — but it is not supported either, and
it should not be treated as established.

## 5. The position rule, and what rules out a proxy cause

**Every invention sits at position 0 of its batch** — or positions 0 and 1 in the two turns that
had two inventions. Never mid-batch, never last, across widths 2, 2, 8, 9, 9, 13 and 28. Nine for
nine.

That is the opposite of what a sampler tail artefact would produce, and it is also what rules out
the most plausible proxy bug. `runtool_call` *looks* like delta-boundary concatenation, so it was
checked directly:

* `_TOOL_MARKER_RE`'s latch sets `held = True`, which **discards** held text. A marker split across
  deltas loses a fragment; it cannot be prepended to a name.
* A bare `run` never appears as a name anywhere in the session, so nothing partial was available to
  fuse.
* `run_shell_command` parsed intact **65 times**, including five in the same batch as
  `runtool_call`. A slicing fault on `run_*` would have corrupted some of those.
* `normalize_tool_call_dict` returns `None` unless `d["name"]` exists, and all six `add_call` sites
  take the name verbatim from upstream text (DSML `name=`, JSON `name`/`function`/`tool`/`action`,
  `<function=NAME>`, bracket form). A `{"arguments": {…}}` object yields **no call at all**, so the
  proxy cannot manufacture `arguments` or `name` as a name.

Position 0 is also the slot that stresses the streaming path hardest — the latch arms on the first
opener it sees, `_HOLDBACK = 32` keeps a tail open in case a marker is split, `_MARKER_PREFIX_LIMIT
= 8` guards short prefixes. A degenerate stub as call #0 therefore exercises exactly the code §2.11
hardened.

One thing this session checked and **cleared**: a text part in the docs phase read
``PDATE.md` already exists first.` — a missing leading `U`, which would be the signature of lost
text on the streaming path. It is not. Concatenating the record's non-thought parts gives
``...whether an `UPDATE.md` already exists first.`` — the `U` ends the preceding *content* part and
the next content part continues after an interleaved *thought* part. The seam is the client's
part-splitting, and no text is missing. Worth recording because the shape looks exactly like a
fault, and because the holdback is per-field and contiguous (§2.11), so it cannot drop a character
from the middle of a string by construction — only a whole unflushed tail.

## 6. What predicts an invention: width holds weakly, depth is untested

Final totals: **150 calls, 9 inventions = 6.0 %**, across 33 turns.

| context | calls | invented | rate |
|---|---|---|---|
| ~9 % | 55 | 4 | 7.0 % |
| — | 87 | 5 | 5.7 % |
| — | 102 | 7 | 6.9 % |
| **11.5 %** | **113** | **7** | **6.19 %** |
| end of session | 150 | 9 | 6.0 % |

The rate is a poor guide, because the denominator grows with every narrow verification turn. The
sharper cut is by batch width:

| | turns | with an invention |
|---|---|---|
| single-call (width 1) | **11** | **0** |
| multi-call (width ≥ 2) | 22 | 7 |

**No invention has ever occurred in a single-call turn.** That is the most robust pattern here:
a turn that emits exactly one call is clean, 11 times out of 11. Widths that invented:
`2, 2, 8, 9, 9, 13, 28`. Widths that did not: `1` ×11, `2` ×8, `3` ×3, `4` ×2, `8`, `27`.

But the correlation is **weak, not clean**, and the last invention is what exposes it. The ninth
`call` arrived in a batch of **width 2** (§11) — as narrow as an invention has ever been, in a phase
whose maximum width was 4. If width were the mechanism, that turn should have been clean. It was
not.

So the honest statement is:

* **Width 1 is safe**, consistently and without exception in this session.
* **Width alone does not decide it.** Two width-2 turns invented and eight did not; one width-8 turn
  invented (as did 9, 13, 28) and another did not.
* **Context depth remains untested.** The wide batches happened early, in the audit phase, because a
  code audit opens with a broad read; the narrow phase came later with a *larger* context and still
  produced an invention. That ordering is the opposite of what a depth story predicts, but one
  sample cannot settle it — §9 lists what would.

## 7. What did not happen

These are the results that make the proxy look healthy, and each is a counter that could have moved:

* **Zero leaked `<tool_call>` markup** in any client-visible text field. This is §2.9's fatal path —
  the block reaching the client, which raises `malformed tool call` and kills the turn. It stayed at
  0 across 51 completions, including the 28-call batch.
* **Zero `repetition` / `unparsed` / `malformed` lines** in the journal. §2.11's fix had nothing to
  do — which is a success, not a gap: the condition it guards simply did not arise.
* **Zero non-2xx** from upstream. The `500` that preceded the original loop never recurred, so the
  precondition for that failure was absent.
* **Zero `api_error` records** in the transcript.
* **The 28-call batch decoded intact**, junk stub at #0 included, with all 27 legitimate calls
  dispatched and the turn surviving. Before §2.11, a malformed opener in that position is what
  produced the original abort.
* **The model declined to invent when asked to.** A live request (2026-09-26, after the session)
  declaring only `read_file` and asking the model to run `ls` produced
  `"I'm sorry, but I don't have a tool available to execute shell commands like \`ls\`"` with
  `finish_reason: stop` — no invented name and no empty call. The nine inventions came from the
  model's own tool choice mid-task, not from a prompt naming a tool that does not exist.

## 8. Three content errors, which are a different class

The audit found and fixed defects in its *own* work. They are recorded separately because they are
not proxy faults and were not caused by the model mis-calling a tool:

| time | what | caught by |
|---|---|---|
| 20:44:11 | `AssertionError` — the intended `edit`s never applied | the model's own verification script |
| 20:44:47 | diagnosed from `git diff`: only `.gitignore` and `LICENSE` had changed | re-read of `git status` |
| 20:46:12 | a partial `edit` left a stray `class NMRV…` fragment | the `edit` tool echoing the modified region |

The first is the notable one: five `edit` calls in the 28-call batch returned **OK** and changed
nothing. Had the model trusted those results instead of asserting, it would have reported the
cleanup as done. That is the argument for the model writing its own verification, and for the
`edit` tool returning the modified region rather than a bare success. All three were resolved: the
re-issued edits applied and the test suite passed (§12).

Also benign, recorded so they are not re-diagnosed as faults:

* `read_file` on `~/.qwen/projects/-home-logan-software-ndlite/memory/MEMORY.md` →
  `file_not_found`. The project had no `memory/` directory; eight other projects on this machine do.
  The model was reading its own project memory on startup, which is correct behaviour.
* Two `run_shell_command` results with `shell_execute_error` and **full output**: `diff` of two
  differing files, and a `grep -q`. Both exit 1 *because* the command did its job. `classify()`
  only exempts `status == "cancelled"`, so this class is reported; see §10.

## 9. Revisit checklist — worked 2026-09-26

1. ~~Re-run the §2 aggregates on the final transcript.~~ **Done.** 418 records, 33 turns, 150 calls;
   figures in §1 and §6 are final. The transcript's last write is `23:06:40Z`.
2. ~~Check whether any invention landed at a position other than 0/1.~~ **Done — the rule holds.**
   Nine for nine. The ninth invention (23:03:13) is at position 0 of a width-2 batch, exactly as the
   rule predicts.
3. ~~Check whether any invented call carried non-empty arguments.~~ **Done — none did.** All nine
   were `{}`. The failure mode has been uniformly empty-argument.
4. ~~Check whether a name from §4's predicted list appeared.~~ **Done — none did.** `type`, `id`,
   `function`, `parameters`, bare `run`: zero occurrences in this session and in every transcript
   under `~/.qwen/projects`. §4 now records this as unsupported rather than pending.
5. ~~Read the journal tail for `repetition` / `unparsed` / `malformed`.~~ **Done — still zero.**
   51 × `200 OK`, 9 warnings, all undeclared-name.
6. ~~Ask whether the user ran tests or committed in `~/software/ndlite`.~~ **Done — see §12.** The
   session committed three times and pushed; the test suite passed.

One question the checklist did **not** anticipate, and which §11 answers: whether resuming a session
after a long idle period changes anything. It does not change the position rule, but it does
disprove the tidy width-only story.

## 10. Instrumentation gaps

*   ~~`TABBY_PROXY_RAW_LOG` was not set.~~ **Armed 2026-09-26** in `~/.config/tabby-proxy.env` (with
    a backup of the file beside it), so the next probe captures invented payloads at the wire without
    needing to be set up in advance. Note the earlier observation: it is read at import (line 512),
    so arming it needs a restart, and `./install.sh proxy` will not rewrite the file. Verified live
    rather than assumed — asking the model for an empty `arguments` object produced both lines:

    ```
    [WARNING] Tool call read_file is missing required parameter(s) ['file_path']; emitted keys were []
    [WARNING] Raw tool-call payload: '<tool_call>\n{"name": "read_file", "arguments": {}}\n</tool_call>'
    ```

    That payload is a *declared* name, so it does not answer the invented-name question — it proves
    the mechanism fires and shows the wire format, which is what a future probe needs.
*   ~~**Batch width is not recorded anywhere.**~~ **Corrected — it was, and now it is in the warning
    too.** The width was always available as the `N` in `Intercepted and parsed N tool call(s)` (and
    the ordered name list on the same line), so §6's correlation was already answerable from the
    journal — this section overstated the gap. As of 2026-09-26 the undeclared-name warning also
    carries `batch=<n>`, so the invented-name row and the width are one row instead of two that must
    be joined by timestamp. What remains genuinely unrecorded is the *depth*, below.
*   **Context depth is not visible to the observer at all.** The 115 000-token figure came from the
    user reading the client UI. Nothing in the transcript or the journal reports it, so any depth
    correlation has to be assembled by hand from user-supplied readings. The transcript's byte size
    is not a proxy for it — it was 984 KB at 115 k tokens but holds every tool result verbatim, and
    those are re-sent each turn. This is the one instrumentation gap that §13's item 3 cannot work
    around.
*   `tabby_watch.py` reports a nonzero-exit `run_shell_command` as `[TOOL_ERROR]` with no way to
    tell "the tool broke" from "the exit code is the answer" (`diff`, `grep -q`, `assert`). The
    output body's presence is an imperfect discriminator; `assert` fails with no output and *should*
    be reported, while `diff` fails with full output and should not.

## 11. The unwatched tail, and the ninth invention

The watcher was stopped by request before the docs phase, so the last two sections come from
re-reading the transcript afterwards. The tail contains one finding and one structural surprise.

**The break.** 132 minutes separate the last audit-phase turn (`20:48:57Z`) from the first
docs-phase turn (`23:01:55Z`). The user resumed the session with "write a UPDATE.md file with the
changes you made and rationale just in case we revisit later" — the model had not been idle-looping,
it had been waiting on the user. Nothing in the journal shows activity in the gap.

**The ninth invention** arrived in that resumed phase, at `23:03:13Z`, in a batch of **width 2**:

```
[0] TEXT thought  'The user wants me to write an UPDATE.md file documenting the changes…'
[1] TEXT          "I'll write `UPDATE.md` recording this cleanup session…"
[2] TEXT thought  'project memory. Let me proceed.'
[3] TEXT          'PDATE.md` already exists first.'
[4] CALL call_ef810ad7  'call'  {}
[5] CALL call_9808de80  'glob'  {"pattern": "UPDATE.md"}
```

It is the *same* name as the audit-phase stub, 2 h 15 min later, again at position 0, again empty.
Then the phase continues normally: `write_file`, `git commit`, `git push`, and a README badge fix,
with 13 further calls and no further invention.

**Why this matters for §6.** The docs phase ran with a *larger* context than the audit phase and its
widest batch was 4 — yet it still produced an invention. Width 2 is the narrowest batch an invention
has appeared in. That is evidence against a width-only mechanism and, because it happened at the
largest context of the session, weak evidence *for* a depth contribution. With one sample it settles
nothing; it does mean §6's original confidence was too high.

## 12. The audit's outcome

Separate from tool-call behaviour, the session did the job it was given. Three commits, authored
`21tesla`, pushed:

```
b7370fd  docs: align Python version badge with requires-python >=3.10
e047a20  docs: record 2026-09-26 cleanup rationale in UPDATE.md
f34d4f4  chore: remove duplicate SettingsDialog and dead imports; fix peak-load and
         remove_spectrum index; add LICENSE
```

`f34d4f4` is the substantive one: `main_window.py` −127 lines net, plus `io_controller.py` and
`peak_controller.py` fixes, a new `LICENSE`, a `.gitignore` entry, and a `tests/test_overlay.py`
adjustment — 12 files, +48/−250. The duplicate `SettingsDialog` is gone from `main_window.py`
(`grep -c` returns 0 there and 1 in `dialogs.py`, which is where it belongs).

Verified independently of the session: `git status` is clean, all three files (`UPDATE.md`,
`LICENSE`, `.gitignore`) are tracked, and the working tree matches `b7370fd`.

So the invented names cost the audit **nothing**. Nine bad calls out of 150, every one discarded as
a recoverable error, and the work still landed and was pushed. That is the practical answer to
whether this failure mode needs a fix: at this rate it is noise the client absorbs, not a hazard.

## 13. What would settle the open questions

The two unresolved questions are §6's mechanism and §11's depth contribution. Both need the same
instrument, so they should be settled together. **Items 1 and 2 are done as of 2026-09-26** — armed
and shipped, waiting on the next session:

1. ~~**Arm `TABBY_PROXY_RAW_LOG=1` before the session starts.**~~ **Done** — set in
   `~/.config/tabby-proxy.env`, proxy restarted, flag confirmed present in the running process's
   environment and confirmed live by a request that produced a raw dump. It captures the invented
   payload at the wire, which is the only way to know whether the stub is a bare `<tool_call>` with
   nothing in it or carries a partial JSON body the extractor dropped.
2. ~~**Add `batch=<n>` to the undeclared-name warning.**~~ **Done** — shipped, with self-test cases
   for width 1 and width 3. §6 is now a journal query rather than transcript archaeology.
3. **Run two sessions with the same batch profile at different context depths** — e.g. the same
   "read these N files in parallel" opening at ~10 k and at ~200 k tokens. If the stub rate is the
   same, it is width; if it rises, it is depth. This session had them confounded in one direction
   and gave an ambiguous hint in the other. *Still the only way to settle §6; note that no
   instrumentation can supply the depth reading — it has to come from the client UI.*
4. **Watch for a non-empty invented call.** All nine were `{}`. A stub with arguments would be a
   different failure — the model would have had a target in mind — and it would change which
   code path is worth hardening. *Item 1's armed dump now makes this decidable on first sight.*

An incidental result of building this: the model **declines** rather than inventing when asked
directly. A request to run `ls` with only `read_file` declared produced
`"I'm sorry, but I don't have a tool available to execute shell commands like \`ls\`"` and a plain
`stop` — no invented name. Nine for nine inventions came from the model's *own* choice of tool
during real work; it does not manufacture one on demand. Reproducing the failure therefore needs a
genuine task, not a prompt that names a fake tool.
