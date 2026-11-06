# fm-shim

A small proxy in front of Apple's `fm serve` (macOS 27.2+) that rewrites the
requests it rejects, so OpenAI-compatible agent clients can use Apple's
Foundation Models, both on-device (`system`) and Private Cloud Compute (`pcc`).

```
your client ──> fm-shim :1976 ──> fm serve :1977 ──> system / pcc
```

## Why

`fm serve` decodes requests with Swift's `JSONDecoder`. When a request has a
shape it can't model, it answers with HTTP 400 in about 10 ms, before any
inference:

```
Invalid request: The data couldn’t be read because it isn’t in the correct format.
```

Most agent clients trip over at least one of these:

| `fm serve` rejects | fm-shim sends instead |
|---|---|
| a tool schema with `anyOf` / `oneOf` | the first non-null branch (keeping the parent's `description`, `title`, `default`) |
| a tool schema with `"type": [...]` | the first non-null type |
| a `developer` message | a `system` message |
| `reasoning_effort` `minimal` on `pcc` | `low` |
| any other `reasoning_effort` it doesn't accept (anything on `system`; `none`, `xhigh`, etc. on `pcc`) | the field is dropped |

Everything else passes through untouched, and responses (including streams)
are relayed as they arrive.

**Collapsing a union loses its other branches.** For example, a parameter that
accepts `true` *or* a list of patterns becomes boolean-only. The model can still
call the tool; it just can't use the dropped forms.

## Install

```sh
cargo install --git https://github.com/aiordanescu/fm-shim
```

or build from a clone with `cargo build --release` (the binary lands in
`target/release/fm-shim`).

## Run

Start both from **Terminal.app**. `fm serve` only offers `pcc` when Terminal
started it; from launchd, ssh, IDE terminals or other apps it reports
"Private Cloud Compute is not available in this context." `system` works from
anywhere.

```sh
fm serve --port 1977 &
fm-shim --listen 1976 --upstream 1977
```

Then point your client at `http://127.0.0.1:1976/v1`, model `system` or `pcc`,
with any API key.

```
fm-shim [--listen PORT] [--upstream PORT] [--log FILE] [--verbose]

  --listen PORT     port to listen on, 127.0.0.1 only (default 1976)
  --upstream PORT   port `fm serve` listens on, 127.0.0.1 (default 1977)
  --log FILE        append the request log to FILE (default: stderr)
  --verbose         also log each individual rewrite
```

The log has one line per request and one per response:

```
14:19:38 model=pcc stream=True msgs=2 last_roles=['system', 'user'] tools=20 rewrites=3
14:19:44   -> 200 first_byte=0.0s total=6.0s bytes=1031
```

Check it's up with `curl -s localhost:1976/health`; with `pcc` available you'll
see `"available": true` for both models.

## Things to know about `fm serve`

- Context windows: 32K tokens on `pcc`, 4K on `system`. Large agent prompts
  only fit on `pcc`.
- `reasoning_effort` works only on `pcc`, and only `low`, `medium`, `high`.
- `pcc` is rate-limited per Apple Account. Over the quota, `fm serve` returns
  HTTP 429 `insufficient_quota` with `Retry-After: 900`. Check it with
  `fm quota-usage`.
- Send `"stream": false` explicitly if you want a plain JSON response.

## Not handled

`fm serve` also rejects `allOf`, `$ref` / `$defs`, `prefixItems`, arrays
without `items`, and schema-valued `additionalProperties`. fm-shim doesn't
rewrite those yet; tools from MCP servers sometimes use them.

## License

MIT
