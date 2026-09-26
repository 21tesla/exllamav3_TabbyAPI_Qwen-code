# SPECIFIC.md — the 2026-09-26 `ndlite` probe

One observed session, recorded in full because it is the largest body of evidence on what this
checkpoint actually does wrong when it emits tool calls. **Everything below is measured from that
session**, not inferred: a code audit of `~/software/ndlite` run on
`DeepSeek-V4-Flash-0731-exl3-2.32bpw` through the proxy, deliberately pushed through a large
context while `tabby_watch.py` tailed both the transcript and the proxy journal.

Session: `7edd9e8c-aa41-47a4-9bfb-479a354fb2fd`
Transcript: `~/.qwen/projects/-home-logan-software-ndlite/chats/7edd9e8c-*.jsonl`
Window: `2026-09-26T20:30:13Z` → `20:49:27Z` (19.2 min, 335 records)
Context reached: ~115 000 tokens (11.5 % of 1 M) at ~984 KB of transcript

The session was still running when these notes were written. §9 is the checklist for finishing it.

## 1. Headline

| | |
|---|---|
| Assistant turns carrying calls | 23 |
| Tool calls emitted | 130 |
| Calls naming a tool that does not exist | **8 = 6.15 %** |
| Calls whose client-visible text leaked `<tool_call>` markup | **0** |
| Upstream non-2xx responses | **0** |
| Proxy `[WARNING]` lines | 8 — all of them the undeclared-name check |
| Journal hits for `repetition` / `unparsed` / `malformed` | **0** |
| Turns aborted by the client | **0** |

Every one of the eight bad calls was absorbed by the client as a recoverable
`tool_not_registered` result, and the turn continued. The failure that motivated §2.11 — a payload
with no parseable call at all — did **not** recur, on 32 upstream completions.

## 2. Method, so it can be repeated

```bash
# start the observer (it is a resident monitor; wind it down deliberately)
~/software/exllamav3-anemone/venv/bin/python \
  ~/software/exllamav3_TabbyAPI_Qwen-code/tabby_watch.py

# the proxy's side, which transcripts do not record
journalctl --user -u tabby-proxy@home-logan.service --since "16:30" --until "16:50" -o cat
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

## 3. The eight inventions

| time (Z) | batch width | position | name | args |
|---|---|---|---|---|
| 20:32:36 | 9 | 0 | `tool_result` | `{}` |
| 20:36:12 | 8 | 0 | `on` | `{}` |
| 20:41:30 | 2 | 0 | `arguments` | `{}` |
| 20:41:30 | 2 | 1 | `name` | `{}` |
| 20:44:06 | 28 | 0 | `call` | `{}` |
| 20:45:32 | 13 | 0 | `ead` | `{}` |
| 20:45:32 | 13 | 1 | `editing` | `{}` |
| 20:48:25 | 9 | 0 | `runtool_call` | `{}` |

All eight carried empty arguments and all eight drew the same client error, verbatim:

```
Tool "X" not found in registry. Tools must use the exact names that are registered.
Did you mean one of: "…"?
```

Proxy side, for each of them, the same two lines in this order:

```
[WARNING] Tool call 'X' is not one of the 20 tools this request declared; the client cannot dispatch it
[INFO]    Intercepted and parsed N tool call(s) from content: ['X', 'read_file', …]
```

That ordering is the diagnostic: the payload **parsed cleanly onto a name the client rejected**.
The separator between this and a parser fault is that the invented string is present in the
`Intercepted and parsed` list — it entered the extractor as that name rather than being assembled
from fragments.

## 4. Three families

The names are not random. They fall into three groups, which is the detail §2.8's phrase
"one-off names" does not capture:

**Wire-format vocabulary** — `tool_result`, `arguments`, `name`, `call`. These are the field names
and content-block types of the OpenAI/Anthropic tool-calling formats. `tool_result` is the
Anthropic content-block type for tool output; `arguments`/`name` are the keys of a call object;
`call_` is the client's own id prefix. The model is emitting the *protocol's own words* in the name
slot.

**Prose and character debris** — `on`, `ead`, `editing`. Fragments of nearby English
("...is **on** the...", "Now I need to **edit**ing"), and in `ead` a truncation missing its first
character. `ead` is the only one that could have been a slicing artefact, and §5 rules that out.

**Fused keyword pair** — `runtool_call`, once. `run` (from `run_shell_command`) and `tool_call`
(the grammar's own tag) collapsed into one name. This is the most informative of the eight: both
halves are vocabulary that was in flight in that batch, and the model merged them rather than
choosing.

**Predicted, not yet observed:** `type`, `id`, `function`, `parameters`, and a bare `run`. If the
wire-vocabulary family is real, those should appear before `skills`-style inventions recur.

## 5. The position rule, and what rules out a proxy cause

**Every invention sits at position 0 of its batch** — or, in the two turns with two inventions,
positions 0 and 1. Never mid-batch, never last, across widths 2, 8, 9, 9, 13 and 28.

That is the opposite of what a sampler tail artefact would produce, and it is also what rules out
the most plausible proxy bug. `runtool_call` *looks* like delta-boundary concatenation, so it was
checked directly:

* `_TOOL_MARKER_RE`'s latch sets `held = True`, which **discards** held text. A marker split across
  deltas loses a fragment; it cannot be prepended to a name.
* A bare `run` never appears as a name anywhere in the session, so nothing partial was available to
  fuse.
* `run_shell_command` parsed intact **52 times**, including five in the same batch as
  `runtool_call`. A slicing fault on `run_*` would have corrupted some of those.
* `normalize_tool_call_dict` returns `None` unless `d["name"]` exists, and all six `add_call` sites
  take the name verbatim from upstream text (DSML `name=`, JSON `name`/`function`/`tool`/`action`,
  `<function=NAME>`, bracket form). A `{"arguments": {…}}` object yields **no call at all**, so the
  proxy cannot manufacture `arguments` or `name` as a name.

Position 0 is also the slot that stresses the streaming path hardest — the latch arms on the first
opener it sees, `_HOLDBACK = 32` keeps a tail open in case a marker is split, `_MARKER_PREFIX_LIMIT
= 8` guards short prefixes. A degenerate stub as call #0 therefore exercises exactly the code §2.11
hardened.

## 6. What predicts an invention: batch width, not context depth

| context | calls | invented | rate |
|---|---|---|---|
| ~9 % | 55 | 4 | 7.0 % |
| — | 87 | 5 | 5.7 % |
| — | 102 | 7 | 6.9 % |
| **11.5 %** | **113** | **7** | **6.19 %** |
| end of window | 130 | 8 | 6.15 % |

The rate is a poor guide, because the denominator grows with every narrow verification turn. The
turns that carry an invention had widths **9, 8, 2, 28, 13, 9**; the session also contains eight
single-call turns and several of width 2–3, and **none of those invented anything**. An invention in
a single-call turn was never observed.

So the working hypothesis is: **wide parallel batches produce the stub; context depth has not been
shown to matter.** Confounding the two is easy — the wide batches happened late in the session
because a code audit starts with a broad read and ends with narrow edits. This session cannot
separate them, and §9 lists what would.

## 7. What did not happen

These are the results that make the proxy look healthy, and each is a counter that could have moved:

* **Zero leaked `<tool_call>` markup** in any client-visible text field. This is §2.9's fatal path —
  the block reaching the client, which raises `malformed tool call` and kills the turn. It stayed at
  0 across 32 completions, including the 28-call batch.
* **Zero `repetition` / `unparsed` / `malformed` lines** in the journal. §2.11's fix had nothing to
  do — which is a success, not a gap: the condition it guards simply did not arise.
* **Zero non-2xx** from upstream. The `500` that preceded the original loop never recurred, so the
  precondition for that failure was absent.
* **The 28-call batch decoded intact**, junk stub at #0 included, with all 27 legitimate calls
  dispatched and the turn surviving. Before §2.11, a malformed opener in that position is what
  produced the original abort.

## 8. Three content errors, which are a different class

The audit found and fixed three defects in its *own* work. They are recorded separately because they
are not proxy faults and were not caused by the model mis-calling a tool:

| time | what | caught by |
|---|---|---|
| 20:44:11 | `AssertionError` — the intended `edit`s never applied | the model's own verification script |
| 20:44:47 | diagnosed from `git diff`: only `.gitignore` and `LICENSE` had changed | re-read of `git status` |
| 20:46:12 | a partial `edit` left a stray `class NMRV…` fragment | the `edit` tool echoing the modified region |

The first is the notable one: five `edit` calls in the 28-call batch returned **OK** and changed
nothing. Had the model trusted those results instead of asserting, it would have reported the
cleanup as done. That is the argument for the model writing its own verification, and for the `edit`
tool returning the modified region rather than a bare success.

Also benign, recorded so they are not re-diagnosed as faults:

* `read_file` on `~/.qwen/projects/-home-logan-software-ndlite/memory/MEMORY.md` →
  `file_not_found`. The project had no `memory/` directory; eight other projects on this machine do.
  The model was reading its own project memory on startup, which is correct behaviour.
* Two `run_shell_command` results with `shell_execute_error` and **full output**: `diff` of two
  differing files, and a `grep -q`. Both exit 1 *because* the command did its job. `classify()` only
  exempts `status == "cancelled"`, so this class is reported; see §10.

## 9. Revisit checklist when the session ends

1. Re-run the §2 aggregates on the final transcript — the window above stops at 20:49:27Z.
2. Count inventions again and check whether any landed at a position **other than 0/1**. One such
   case would falsify §5's position rule.
3. Check whether any invented call carried **non-empty** arguments. All eight so far were `{}`; a
   non-empty one is a different failure and would change the assessment.
4. Check whether a name from §4's predicted list appeared (`type`, `id`, `function`, `parameters`,
   bare `run`).
5. Read the tail of the journal for `repetition` / `unparsed` / `malformed` once more — the probe
   ended before the session did.
6. Ask whether the user ever ran the test suite or committed in `~/software/ndlite`; the audit's
   *outcome* is a separate question from its tool-call behaviour.

## 10. Instrumentation gaps

*   `TABBY_PROXY_RAW_LOG` was **not** set, so none of the eight payloads was captured at the wire.
    §2.8 is explicit that this is the only way to answer whether an invented call carried meaningful
    arguments, and here the emptiness is inferred from the extraction rather than observed. Arming
    it needs a restart — `_RAW_LOG` is read at import (line 512) — which was deliberately not done
    mid-session. `TABBY_PROXY_RAW_LOG_CHARS` (20000) already bounds the dump.
*   **Batch width is not recorded anywhere.** It is derivable from the transcript but not emitted by
    the proxy, so the width-vs-depth question in §6 cannot be settled without post-hoc transcript
    work. A one-field addition to the undeclared-name warning (`batch=<n>`) would make it a
    journal-side measurement.
*   **Context depth is not visible to the observer at all.** The 115 000-token figure came from the
    user reading the client UI. Nothing in the transcript or the journal reports it, so any
    depth correlation has to be assembled by hand from user-supplied readings.
*   `tabby_watch.py` reports a nonzero-exit `run_shell_command` as `[TOOL_ERROR]` with no way to
    tell "the tool broke" from "the exit code is the answer" (`diff`, `grep -q`, `assert`). The
    output body's presence is an imperfect discriminator; `assert` fails with no output and *should*
    be reported, while `diff` fails with full output and should not.
