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
"one-off names" does not capture. **Read this section together with §14.1**: the second probe
proved that two of these families (`ead`-style truncation and the wire-format vocabulary) are
produced by our *own* tag scanner misreading a mangled closer, and that the grouping below is
about what the model emitted, not about what the client was told.

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
the most plausible proxy bug. **§14.1 retires this rule as evidence**, at least for the
DSML-sourced names: `extract_tool_calls` inserts the DSML pass's results ahead of the JSON pass's
unconditionally, so a stray tag appearing 8th in the stream is still reported at index 0. The
reasoning below therefore stands only for calls that came from the JSON scan. `runtool_call`
*looks* like delta-boundary concatenation, so it was checked directly:

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

## 14. The second probe — same session, longer context, different failure (2026-09-26 19:00→19:30)

The ndlite numbers above are the **first** probe. A second ran later the same day in
`e8ee13a8` (the very session §2.11's original abort came from), on the same model, with
`TABBY_PROXY_RAW_LOG` armed. It changed the thesis, so it is recorded separately: the ndlite
figures must not absorb these.

Totals for the day, all sessions, from the proxy journal: **316 calls, 13 undeclared-name
warnings, 2 dropped-required-parameter warnings, 3 unparsed completions, 0 repetition, 0
malformed, 0 non-2xx on inference.** Twelve of the thirteen undeclared names are in this one
session, against nine in the whole ndlite audit.

### 14.1 The instrument earned its keep immediately

Arming the raw dump was the difference between a guess and a diagnosis. The two new warnings
looked, from the decoded line alone, like invented tool names:

```
19:16:11 [WARNING] Tool call '_call'        is not one of the 20 tools this request declared (batch=4)
19:19:36 [WARNING] Tool call '_placeholder' is not one of the 20 tools this request declared (batch=4)
```

`batch=4` is the item 2 instrumentation working in production. The dumps, replayed through the
proxy's **own** `extract_tool_calls` offline, showed neither name is invented by the model:

| payload | the model wrote | our scanner read |
|---|---|---|
| `19:16:11` | `<tool_call>{grep_search}</…>` then a stray closer `</｜DSML｜_call>` | a call named `_call`, `{}` |
| `19:19:36` | `<` `\` `｜DSML｜_placeholder>` — meant `</`, wrote a backslash | a call named `_placeholder`, `{}` |

Neither payload contains a `"name": "_call"`. Both artefacts are **our parser**, and the
mechanism is specific: `_TAG_RE = <[^<>]{0,200}>` matches `</｜DSML｜_call>` because the bar is
the fullwidth U+FF5C, not ASCII. `split_dsml_tag` folds each bar to a space and takes the first
word after DSML — `_call` — and `classify_dsml_tag` treats *any* DSML tag carrying a keyword as
the name-as-tag-name dialect, so a **closer** became a call.

This reinterprets §4 and §5 rather than extending them:

* `runtool_call` is not unique. The family is **characters lost at long context**, and `_call`,
  `_placeholder` and probably `all`, `ead`, `editing`, `on` (substrings of tag names) are in it.
  Only `_call` and `_placeholder` are *proven*, because only they were inside an armed dump.
* **§5's position-0 rule is an artefact of our own step ordering.** `extract_tool_calls` runs the
  DSML pass (step 1) before the JSON pass (step 2), so every DSML-sourced call is inserted ahead
  of every JSON-sourced one. In the `_call` payload the stray tag is the **8th** tag in the
  stream, and it still lands at index 0. Position 0 was offered as evidence against a sampler
  tail; for DSML-sourced calls it carries no such signal. The `runtool_call` position argument
  stands only if that artefact was JSON-sourced.
* §2's method gains its strongest tool: `ast.literal_eval` on the log's `repr()`, then replay
  through the real extractor. A guess about parser behaviour is worth nothing next to it.

### 14.2 The width story does not survive

The same window produced the widest batch of either probe:

```
19:29:10  parsed 48 tool call(s) from content: 25 grep_search + 15 glob + 8 read_file
```

**Width 48, and it was entirely clean** — no stub, no undeclared name, no dropped parameter.
§6 leant on "width ≥ 2 is where risk lives". The width-1 safety claim is untouched (still no
invention in a width-1 turn anywhere), but width is now a poor predictor in the other direction:
one clean 48, clean 2s, dirty 2s. What the dirty turns share is not width but **malformed
emission** — and once the parser artefacts are excluded (§14.1), the remaining failures are all
of one kind.

### 14.3 The real fault: ASCII glyph loss past ~180k tokens, and the session it cost

The session ended at **195 432 input tokens** (`context` at 186 000 per the user). Its last
completion took 37.8 s and came back with the model's own glyphs corrupted:

```
23:29:51  in=195432 out=2046 status=200 dur=37843ms
```

The payload holds **1 opening `<tool_call>` and zero `</tool_call>`**; every closer is a
mangled `</｜tool>`-shaped fragment. The mangling is **exact and locatable**: 98 ASCII quotes,
then at raw offset 3949 — precisely the first mangled closer — everything following uses
typographic `“ ”` (138) and fullwidth `｜` (36). A `{"name": "write_file"}` with a curly quote is
not JSON; `raw_decode` failed, `repair_truncated_json` returned None, both the DSML and JSON
passes found nothing, and `extract_tool_calls` returned `None`.

What followed, measured on the captured payload:

| reading | recovered |
|---|---|
| as shipped | **0 of 7 calls** |
| quotes folded to ASCII first | 6 of 7 (all after the first) |
| truncate at the first mangled closer | the first (`write_file`) |

So two independent defects, and the second is not about glyphs at all. The **first** call's
object ends `…main())\n"}` with **zero curly quotes in it** — it was already ASCII. It ends
`</｜DSML｜…>` instead of `}`, so its own closer is missing. `repair_truncated_json` exists for
exactly that and handles it when handed the bounded fragment; the caller handed it *the rest of
the completion*, so its `cuts` search found the last structural character of the whole payload,
placed all the later markup inside the fragment, and failed. The one call that was ASCII was
lost by a bounding bug in the caller, not by the glyph loss.

Downstream, the turn did not reach the client as an error. `process_message_tools_and_thinking`
downgraded `finish_reason` to `stop` and handed over 5 525 characters of prose with
`<tool_call>` and `{“name”: …` in it. The safety net §2.8 added did its job **in this window** —
no `InvalidStreamError`, no abort — but the model's answer was a *description* of a `write_file`
it never got to perform, as plain text. **§16 qualifies what that net reaches**: it answers one of
the client's three abort conditions, and every abort observed on this checkpoint predates it, so
this clean window is not evidence that the class is closed. That is the observable end of the session: last
inference request 19:29:52, no request after it, and nothing but a local `/stats` in the
transcript.

The same glyph loss also produced the three mangled-closer warnings above, and two `unparsed`
rows earlier in the day from other shapes (a truncated opener at 15:10, and **nine consecutive
`<tool_call>` openers with no closers** at 15:12 — a repetition-like loop that
`_strip_repetition_tail` did not catch, because it only strips a run at the very *tail*).

### 14.4 Fixes shipped, and what they are worth

Three changes, each one replayed against the captured payloads and locked into `--selftest`
(63 → **68** cases):

1. **`fold_typographic`** — a second reading of the completion with `“ ” ‘ ’` mapped to ASCII
   quotes. Applied *per bracket and only after the model's own glyphs fail*, never in place: an
   apostrophe in prose is legitimate text, and a real call must win. Folding is hoisted out of
   the scan, since the completion can be hundreds of KB.
2. **`_fragment_ends` + resuming after a repair** — bounds the fragment offered to
   `repair_truncated_json` at closing-tag positions only. Interior braces are deliberately *not*
   offered: a cut there yields a fragment that parses while having silently dropped part of the
   value, which is the one outcome the repair is written to avoid. The scan also resumes after
   the repaired value instead of jumping to the end of the text, so a malformed first call no
   longer hides its siblings — worth 6 of the 7 calls here.
3. **`fold_dsml_syntax` + `classify_dsml_tag` guard** — normalises the mangled tag forms, and
   refuses to read a DSML tag as a call when it is a *closer* or when its keyword begins with
   `_`. No tool name starts with an underscore; `_call` and `_placeholder` are the tails of
   `tool_call` and `function_call`. A `name=` attribute still settles the dialect cases.

Both captured payloads now extract correctly: `_call` → the 3 real calls and no stub;
`unparsed` → all 7. The 48-call clean batch, the v1–v4 fixtures and the repetition suite are
unchanged, and a 74 KB blob of brace-heavy curly-quoted prose parses in 3 ms.

What the fixes do **not** address is the cause. The proxy cannot stop a checkpoint losing ASCII
past ~180k tokens. It can only stop *misreading* the result — and it now recovers 7 of 7 calls
from the exact payload that previously yielded nothing, which converts a lost turn into a slow
one.

### 14.5 What would settle this

1. **A depth sweep on emission integrity.** The 180k figure is one observation. The `parsed N`
   and warning counts already give the rate; a session at ~50 k, ~120 k and ~200 k on identical
   work would show whether the glyph loss is a threshold or a slope.
2. **A `finish_reason`-only fallback.** When a completion yields no call *and* its text contains
   `<tool_call>`, the turn is already saved by downgrading to `stop`. A stronger option is to
   re-ask once with the payload quoted back; not attempted, because it changes the client
   contract.
3. **Whether the earlier families were also parser-side.** `all`, `ead`, `editing`, `on` and
   `oduct` predate the arming or fell outside a dump window. One hour with the flag on an
   equivalent session, then the same replay, would move them out of inference into evidence.
4. **The repetition stripper's `unparsed` gap** — nine openers with no closers mid-completion is
   not a tail run. Cheap to check whether a mid-text run should be collapsed too.

## 15. A third probe, and the one error the observer caused (2026-09-26 22:42 EDT →)

A third session, `e5e8c7f9` (`analysis-qt6-c7`), ran the same evening as §14 at a **lower** context
(133 k–139 k tokens), against an earlier fix set plus the four that followed it — the native
`delta.tool_calls` merge, the `call_tool` attribute dialect, the declared-name guard, and the
`call` per-argument attributes. Those defects and their fixes are recorded in the commits, not
here: this section exists for one thing only, because it is not a finding about the model.

**The session's only `api_error` is the observer's own.**

```
03:08:22 (UTC)  TypeError:UND_ERR_SOCKET  terminated (cause: UND_ERR_SOCKET: other side closed)
```

| time (EDT) | event |
|---|---|
| 23:06:51 | the client's request reaches the proxy and upstream generation begins |
| 23:06:52 | the observer runs `systemctl --user restart tabby-proxy@home-logan.service` |
| 23:06:52–23:08:22 | uvicorn waits, trying to drain the in-flight generation |
| 23:08:22 | `TimeoutStopUSec` (90 s, the default for this unit) expires: `State 'stop-sigterm' timed out. Killing.` → SIGKILL → the client's socket closes |

The proxy cannot drain a generation of that length (the turn before it had taken 154 s), so a
restart during one kills it, and the client records the closed socket as an `api_error`. The probe
retried, the retry landed, and the next served turn was clean — the cost was one turn. Left
unlabelled, though, that row reads as model behaviour in the same transcript the model is being
studied in, which is exactly the kind of contamination this document exists to prevent.

**The mistake was the precondition, not the restart.** An idle *client* was checked and an idle
*proxy* was not. The transcript's mtime only proves the client is between turns, which is a
different question. **The proxy-side test given here in the first draft — `ss -tnp | grep 8081`
showing an `ESTAB` pair — is wrong**, and was corrected on 09-27 after measuring it: that pair is a
persistent keep-alive socket and sits there while the proxy is idle (constant `2/2` over 40 s with
no request in flight). The signals that work are the journal's last line (a request is open iff its
line lacks its closing `POST … 200 OK`) and proxy CPU time (`ps -o cputime= -p <pid>`, unchanged
over ~6 s means idle) — a streaming response costs real CPU, an idle socket none.

Two follow-ons worth keeping:

1. **A deliberate restart is still the right way to get a parser fix onto the wire** — but it must
   be timed to an idle proxy, and a cut made anyway has to be annotated rather than assumed clean.
   `systemctl restart` gives no way to bound the drain; stopping, waiting for the connection to
   clear on its own, and starting is the graceful form.
2. **A filter can hide the very error being counted.** The first read of this session reported "0
   `api_error`" for the whole run; the field is `event.name == "qwen-code.api_error"`, and matching
   the bare substring `"api_error"` matched nothing. The count was 1 from the moment it happened.
   Count on the parsed field, not on a substring of its name.

## 16. The error class the safety net does not close (2026-09-27)

§14.3 records the `e8ee13a8` glyph-loss turn as ending without an `InvalidStreamError`. That is
true of the turn, and **the generalisation from it is not**: this checkpoint produced eleven such
records, and **nine of them reached the client through this proxy**. What the nine do *not* show
is a net failing — every one of them predates the version that carries the net, and the two that
sit in the same session as §14.3 fall four hours before that section's window. The net is real;
the question is how much of the class it can reach, and the answer is one of the client's three
abort conditions. Both corrections here are to my own earlier reading: a filtered count and a
misread clock.

The instrument first: `tabby_watch.py` tested `event.get("event.name") == "api_error"`, but the
field is namespaced `qwen-code.api_error`. The bare test matched nothing, so **every** api_error
in every transcript was invisible to the observer. A direct scan of all transcripts finds 171 of
them; **11 are `InvalidStreamError`** / "Model response contained a malformed tool call." on this
checkpoint (`DeepSeek-V4-Flash-0731-exl3-2.32bpw`), and **nine of them were proxy-served** — the
two exceptions are the first rows below.

| client-local | session | duration | proxy served it? |
|---|---|---|---|
| 2026-09-24 19:26:22 | `843f2fc7` | 18.4 s | **no — before the proxy's first request (09-25 13:50:41)** |
| 2026-09-24 19:26:58 | `843f2fc7` | 11.6 s | no — as above |
| 2026-09-25 19:39:18 | `0c1c9cb3` | 11.7 s | yes — POST 19:39:07 |
| 2026-09-25 19:39:23 | `0c1c9cb3` | 2.4 s | yes — POST 19:39:20 |
| 2026-09-25 19:39:32 | `0c1c9cb3` | 5.7 s | yes — POST 19:39:27 |
| 2026-09-25 19:51:24 | `decb9d2e` | 4.0 s | yes — POST 19:51:20 |
| 2026-09-25 19:51:44 | `08223956` | 20.0 s | yes — POST 19:51:24 |
| 2026-09-25 19:56:21 | `decb9d2e` | 85.9 s | yes — POST 19:54:55 |
| 2026-09-25 20:18:26 | `0faa6605` | 19.2 s | yes — POST 20:18:07 |
| 2026-09-26 15:10:12 | `e8ee13a8` | 45.0 s | yes — POST 15:09:27 |
| 2026-09-26 15:10:32 | `e8ee13a8` | 6.0 s | yes — POST 15:10:27 |

Provenance is settled by **request start time**, not by the completion time: each client error
stamp minus its own `duration_ms` lands on a proxy `POST` within 0.8 s. The two `e8ee13a8` rows
are at client-local **15:10**, four hours before §14's stated `19:00→19:30` window, so they are
**not** in it — §14.3's clean reading of that window stands. They matter because they are the
latest malformed-call aborts on this checkpoint, 09-25 evening through 09-26 mid-afternoon, and
the proxy serving them was still older than the net (`a3cff8d`, 09-26 16:09).

The trap is worth naming because it caught me twice in the other direction. A *client* timestamp
is local and a *transcript* timestamp is UTC; reading an error stamp off one and querying the
proxy with the other returns an empty window, which looks like evidence of absence rather than a
wrong hour. The check that settles it is one command: the proxy's last `POST … 200 OK` against the
transcript's last record reads `23:11:25-04:00` against `03:11:25Z` — four hours, on this box.

**The mechanism, exactly.** The client aborts on three conditions, not one:

```js
if (choice.finish_reason && (
      toolCallParser.hasInvalidToolCallIndex() ||
      toolCallWithoutName ||
      (choice.finish_reason === "tool_calls" && completedToolCalls.length === 0)
   )) throw new InvalidStreamError("Model response contained a malformed tool call.", "MALFORMED_TOOL_CALL")
```

The §2.8 net covers only the third: it rewrites `finish_reason` to `stop` when the upstream said
`tool_calls` and its own parser found nothing. It cannot reach `hasInvalidToolCallIndex()` or
`toolCallWithoutName`, because those are decided by the client's **streaming** parser from deltas
the proxy has already relayed — and they fire under either finish reason. The tightest evidence
is the second `e8ee13a8` row: the proxy logged `Tool-call syntax present but unparsed` at
`15:10:32.795` and the client raised the error at `15:10:32.796`. One millisecond, from the same
completion: the proxy's offline parse failed *and* the client's live parse had by then seen a
malformed call. A proxy that relays deltas — which this one does, deliberately, so the client is
not silent for a whole generation (§2) — is structurally downstream of the failure it wants to
prevent.

**When the net began to work, and what it has been tested against.** `git log -S` puts the net's
text in `a3cff8d` (2026-09-26 16:09). Every recorded abort on this checkpoint is older than that —
the latest is the `e8ee13a8` pair at 15:10 — so none of the eleven is a test of the net; they are
the class as it behaved before the net existed. Every invocation of "reporting stop …" in the
journal, by contrast, falls on 09-26 ≥ 22:12: eight of them, each turning a `tool_calls`
completion the proxy could not parse into plain prose. No abort follows any of them, and no
malformed-tool-call abort occurs anywhere after 16:09, so the net has never faced a streaming-side
failure at all. That is consistent with it working on the one condition it covers, and says
nothing about the other two.

So the shape of it: the net is real and it works on its own condition. It is **not** a guard
against the error class, and that is an argument from the client's code rather than from a
failure — the two conditions it cannot see are settled by the streaming parser, before the
proxy's end-of-turn correction is even reached. The eleven aborts neither contradict the net nor
exercise it.

**What this reopens.** §8 and §14.4 claim the downgrade converted "a lost turn into a slow one".
That holds for a completion the proxy *can* parse as nothing. It does not hold for a completion
the proxy parses as *malformed* while holding back only 32 characters: the relayed prefix is
already on the wire, so a client abort is decided before any end-of-turn correction can land.
Two candidate directions, neither attempted:

1. **Hold back from the first tool marker**, not the last 32 characters — emit the clean prefix
   and no more until the turn resolves. Removes the incremental-delta win the streaming work
   bought, so it is a real trade, not a free fix.
2. **A local repair in the relay** — the `fold_typographic` work already recovers 6 of 7 calls
   from one payload; applying it to the delta path before the client sees it would answer the
   client's condition rather than the proxy's.

**And the instrument, again.** The `tabby_watch.py` fix is one line — match
`qwen-code.api_error`, the field itself, never a substring of its name. §15's second follow-on was
written about the observer's own filter; this is the same bug, in the observer's own source, found
only because the corrected filter was applied by hand first. The lesson generalises: an observer
that counts by substring will report a clean run and mean nothing by it.

## 17. Round 7, on the same probe: a quoted Python argument object (2026-09-27 23:16)

The third probe's second evening produced a dialect the first two never did, and it defeated the
`call_tool` fix from round 3 on two counts at once. The model wrote **five** `grep_search` calls as

```
<｜DSML｜_skill name="grep_search" args="{'pattern': 'molecule_menu|…', 'path': '/…/main_window.py'}"/>
```

and the proxy emitted **two**, one of them with no arguments at all:

```
23:16:50  [WARNING] Tool call grep_search is missing required parameter(s) ['pattern']; emitted keys were []
23:16:50  [INFO]    Intercepted and parsed 2 tool call(s) from content: ['grep_search', 'grep_search']
```

This is round 3's dialect — the whole argument object riding on the call tag as `args=` — with two
changes that each defeat a different piece of the reader:

1. **The object is wrapped in double quotes, and written as Python.** `_ATTR_RE`'s value class was
   `[^"\x27]*`, which forbids *any* quote inside the value, so the match on
   `args="{'pattern': …}"` ended at the first inner `'` and captured `{` alone. The value class has
   to admit one quote and consume to its matching one; the three spellings it now accepts are
   `key="value"`, `key='value'` and `key={…}` as mutually exclusive arms.
2. **What survives that is a Python literal, not JSON.** `json.loads("{'a': 1}")` fails.
   `_parse_attr_arguments` now falls back to `ast.literal_eval`, which evaluates literals only and
   is reached only after JSON has already failed, so a value that is valid JSON keeps the reading
   it has always had.

A third defect in the same payload is independent of both. The second `<tool_call>` carries
`"\.index\("` in its pattern. `\.` is an escape JSON does not define, so `raw_decode` rejects the
object and the scan resumes *past* it — which also lost a sibling call inside the same block.
Python's literal syntax accepts `\.`, so `repair_truncated_json` now tries `literal_eval` as a last
resort; JSON still wins whenever it can read the text.

| reading | calls recovered |
|---|---|
| as shipped (the journal's own count) | **2**, one with `{}` arguments |
| all three fixes above | **4** |
| and the tag body no longer capped short (§18) | **5** |

The payload contains **five** `grep_search` calls and every one is recoverable from the text: three
ride `_skill` tags and two are plain JSON `<tool_call>` blocks. All five keep `pattern` and `path`
intact, the `\.` preserved. §17 first recorded the fifth as the *model's* loss — a truncated tag —
and that was wrong; §18 replays the bytes and corrects it.

The three fixes ship together (`06ae605`) with three new `round7` self-test cases — one per defect,
and one asserting that a JSON object still wins over the literal reading — taking the suite from
90 to **93 cases**. On the captured payload they take 2 → 4; the fifth needs §18.

**Two more undeclared names, and what the guard is for.** The same evening produced `Dropped
non-call DSML tag 'nest'` (23:20:49) and `'arice'` (23:23:12) — the round-4 guard catching two
more invented wrappers, both harmless because the calls inside them were read from their own JSON
bodies. Every undeclared name this checkpoint has produced is an envelope or a fragment of one,
never a tool the model wished existed; that is the whole reason the guard refuses the tag rather
than the call.

## 18. The tag-length cap that silently ate a call, and the test that could not see it (2026-09-27 00:30)

Round 7 was written for the payload quoted in §17 and it fixed the two defects named there. The
payload carries **five** `grep_search` calls, and after round 7 the build recovered **four**. The
fifth was written off as the model's own loss — a `_skill` tag truncated mid-argument, its
arguments unreadable. That was wrong, and re-reading the captured bytes says so: **every `_skill`
tag in the payload is complete**, its `args="…"/` closed. The call was lost in the *reader*, not
in the model.

What lost it is the scanner's tag regex, not any of the three defects §17 names:

```
_TAG_RE = re.compile(r"<[^<>]{0,200}>")
```

A single tag's body was capped at 200 characters. The cap is there for a reason — without a
bound, a stray `<` in prose runs to a much later `>` and swallows the text between — but the
`_skill` dialect is exactly the shape that outgrows it, because it carries its whole argument
object on the tag. The lengths in this one payload, measured rather than guessed:

| tag | body length | matched by the 202-char ceiling? |
|---|---|---|
| `_skill` carrying `molecule_menu\|…` | 253 | **no** |
| `_skill` carrying `_LABELS_HEAD\|_LABEL_\|…` | 191 | yes |
| `_skill` carrying `Delta\|Shift\|…` | 168 | yes |

So the longest tag — the one whose pattern simply ran longer — was skipped whole, and the call
with it. The other two `_skill` calls were read, which is why the failure looked like "four of
five" rather than a dialect the reader never learned.

`_TAG_BODY_MAX` is now 1000, still a bound. The bound is a real protection; the *number* was the
bug.

**Why the round-7 test could not have caught it.** The test written for that dialect used a
*short* pattern:

```
args="{'pattern': '_LABELS_HEAD|HEADERS', 'path': '/tmp/x.py'}"
```

Its tag is a fraction of 200 characters, so it passed on the old `_TAG_RE` exactly as it passes on
the new one — it tested the *quoting* defect and never touched the length. A fixture that passes
on both the broken and the fixed reader proves nothing about the fix; the regression case added
here uses a pattern long enough that the tag clears 200 (`molecule_menu|add_sequence_act|…`), and
it fails on the old cap with **0 calls** and passes on the new one.

The lesson is the one §16 already taught in another guise: the instrument's own limit reads as a
clean result. A parser that skips a tag too long for its regex reports a *smaller* call count, not
an error, and a test whose fixture never reaches the limit reports success. Both say nothing.

| reading | calls recovered from the §17 payload |
|---|---|
| round 3, as first shipped | 2 (one `{}`) |
| round 7 (`06ae605`) | 4 |
| plus `_TAG_BODY_MAX` | **5 of 5** |

**The client's side of the same class, counted.** The empty-argument call is not only a proxy
warning; it is a tool result the client refuses. Over this one session the client rejected **eight**
calls with `params must have required property '…'`, and every one of them carried `args: {}` —
`read_file` ×4, `glob` ×3, `grep_search` ×1, i.e. an empty-argument call is not specific to one
tool. The proxy warned on **seven** of them (six `read_file`, one `grep_search`); the three `glob`
have no matching warning, because they arrived over the upstream *native* `tool_calls` channel and
its rows are the ones the journal records as "Upstream said tool_calls but no call was parsed;
reporting stop" (22:43:33 and 22:43:38) rather than as a shape warning. One of the seven is
confirmed as one call by id: the proxy's `grep_search`/`['pattern']` warning at `23:16:50.946`
and the client's `call_8baf2ff1 grep_search {}` at `03:16:50.959Z` (UTC), whose result at
`03:16:51.000Z` reads `params must have required property 'pattern'`.

**No recurrence.** Since the round-7 deploy at `23:23:12`, `missing required parameter` in the
journal is **0**. The empty-argument class as recorded here is pre-deploy; whether it is closed
outright is what the probe's next turns test.

## 19. Watching a parser that fails silently (2026-09-27 00:35)

§18's loss went unnoticed not because it was hard to see but because *nothing was watching for it*.
The proxy's raw dump fires only when parsing **fails**; a dialect that parses leaves no record that
it was ever seen. So the instrument records a defect only when the defect wins — the same asymmetry
as §16's substring filter, and the reason a call can vanish while every log line reads clean. Three
gaps were closed, all in the direction of making the invisible visible.

**Native calls were never shape-checked.** `_log_tool_call_shapes` ran on calls scraped from the
text fields and on nothing else. A call the *upstream* parsed — TabbyAPI's own `deepseek_v4` DSML
reader, delivered over `delta.tool_calls` — was emitted straight to the client, unchecked. That is
exactly how the three empty `glob` calls reached the client with no warning (§18): their `{}`
arguments were never compared to the schema because nothing compared them. Both native paths now
call `_log_tool_call_shapes` — the streaming one just before it settles the finish reason, the
non-streaming one beside the empty-call guard. An unusable native call is now as visible as an
unusable scraped one.

**A dialect that parses left no trace.** The `skill`/`call_tool` shape carries its whole argument
object as an attribute, and an INFO line now records each time one is read:

```
Read an argument object carried on the '_skill' tag as args= (2 key(s): ['path', 'pattern'])
```

Without it, the only evidence the shape ever arrived is that a call came out right — and since a
*missing* call just makes the count smaller, the shape could regress to unread with no error at all.
This is the line that turns "five calls, no complaints" into "five calls, three of them read from
the attribute dialect."

**An unreadable tag was unreadable without a word.** A call-shaped opener that `_TAG_RE` cannot
match — the §18 case, a body longer than `_TAG_BODY_MAX` — now warns:

```
A call-shaped tag at offset 0 has no match within 1000 chars (len to next '>' is 1255);
its call is not read. Marker: '<|DSML|skill name="grep_search" args="{\'pattern\': \'xxx…'
```

The detector is deliberately strict: it requires `<`, optional DSML framing, an optional `_`, then a
call keyword (`tool_call`, `call_tool`, `skill`, `invoke`, `calls`). A closing tag, the `Output`-echo
envelope, a `name=`-as-tag-name call and a word in prose all fail it — so it marks precisely the
openers that *should* have parsed, and a warning from it means a call was lost, not that the model
wrote something odd.

All three are covered by self-test cases (97 total), because a guard no one can see fire is no
better than no guard. The principle this section records: **on a mission-critical path, instrument
the success as well as the failure** — the failure log is empty in both the good case and the
invisible-loss case, and only the positive trace tells them apart.

