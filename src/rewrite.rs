//! Request rewriting for Apple's `fm serve`.
//!
//! `fm serve` decodes requests with Swift's `JSONDecoder` and answers HTTP 400
//! "The data couldn't be read because it isn't in the correct format" when a
//! request contains things it can't model. This module fixes the ones that
//! common agent clients send:
//!
//! - tool schemas with `anyOf` / `oneOf` / `"type": [..]` unions are collapsed
//!   to their first non-null branch;
//! - `developer` messages are sent as `system`;
//! - `reasoning_effort` is mapped or dropped: only model `pcc` accepts it, and
//!   only `low` / `medium` / `high` (`minimal` becomes `low`).
//!
//! It also fixes one response: `fm serve` reports a context overflow as a
//! generic server error with no numbers ("The session's transcript exceeded
//! the model's context size."), which clients can't act on. It's rewritten to
//! OpenAI's `context_length_exceeded` error, quoting the model's window, so
//! clients that compress on overflow can recover.

use serde_json::{Map, Value};

/// Context windows `fm serve` enforces, per model.
pub fn context_window(model: &str) -> Option<u32> {
    match model {
        "pcc" => Some(32768),
        "system" => Some(4096),
        _ => None,
    }
}

/// Keys kept from the parent object when a union is replaced by one branch.
const KEEP_FROM_PARENT: [&str; 3] = ["description", "title", "default"];
/// Keys whose values are data, not sub-schemas, so they are never rewritten.
const DATA_KEYS: [&str; 5] = ["enum", "required", "examples", "const", "default"];
const PCC_EFFORTS: [&str; 3] = ["low", "medium", "high"];

/// Result of rewriting a chat-completions request body.
pub struct Rewrite {
    pub body: Vec<u8>,
    /// One-line summary for the log; `None` if the body wasn't a JSON object
    /// and was passed through untouched.
    pub summary: Option<String>,
    /// What was changed, one entry per change.
    pub notes: Vec<String>,
    /// The request's `model`, if it had one.
    pub model: Option<String>,
}

pub fn rewrite(body: &[u8]) -> Rewrite {
    let mut doc: Map<String, Value> = match serde_json::from_slice(body) {
        Ok(Value::Object(m)) => m,
        _ => {
            return Rewrite {
                body: body.to_vec(),
                summary: None,
                notes: Vec::new(),
                model: None,
            };
        }
    };
    let mut notes = Vec::new();

    if let Some(Value::Array(tools)) = doc.get_mut("tools") {
        for tool in tools.iter_mut() {
            let Some(Value::Object(func)) = tool.get_mut("function") else {
                continue;
            };
            let name = func
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string();
            if let Some(params) = func.get_mut("parameters") {
                *params = flatten(params.take(), &name, &mut notes);
            }
        }
    }

    if let Some(Value::Array(messages)) = doc.get_mut("messages") {
        for msg in messages.iter_mut() {
            if msg.get("role").and_then(Value::as_str) == Some("developer") {
                msg["role"] = Value::from("system");
                notes.push("developer role -> system".into());
            }
        }
    }

    let model = doc.get("model").and_then(Value::as_str).map(str::to_owned);
    let is_pcc = model.as_deref() == Some("pcc");
    match doc.get("reasoning_effort") {
        None | Some(Value::Null) => {}
        Some(Value::String(e)) if is_pcc && e == "minimal" => {
            doc.insert("reasoning_effort".into(), Value::from("low"));
            notes.push("reasoning_effort 'minimal' -> 'low'".into());
        }
        Some(Value::String(e)) if is_pcc && PCC_EFFORTS.contains(&e.as_str()) => {}
        Some(other) => {
            notes.push(format!(
                "dropped reasoning_effort={} for model {}",
                repr(other),
                model.as_deref().unwrap_or("None")
            ));
            doc.shift_remove("reasoning_effort");
        }
    }

    let roles: Vec<Option<&str>> = match doc.get("messages") {
        Some(Value::Array(m)) => m
            .iter()
            .map(|m| m.get("role").and_then(Value::as_str))
            .collect(),
        _ => Vec::new(),
    };
    let last_roles = roles[roles.len().saturating_sub(3)..]
        .iter()
        .map(|r| r.map_or("None".to_string(), |r| format!("'{r}'")))
        .collect::<Vec<_>>()
        .join(", ");
    let n_tools = match doc.get("tools") {
        Some(Value::Array(t)) => t.len(),
        _ => 0,
    };
    let stream = match doc.get("stream") {
        Some(Value::Bool(true)) => "True",
        Some(Value::Bool(false)) => "False",
        _ => "None",
    };
    let summary = format!(
        "model={} stream={stream} msgs={} last_roles=[{last_roles}] tools={n_tools} rewrites={}",
        model.as_deref().unwrap_or("None"),
        roles.len(),
        notes.len()
    );

    let body =
        serde_json::to_vec(&Value::Object(doc)).expect("serializing a JSON value can't fail");
    Rewrite {
        body,
        summary: Some(summary),
        notes,
        model,
    }
}

/// If `message` is `fm serve`'s context-overflow error, the OpenAI-style message to send instead.
fn overflow_message(message: &str, model: Option<&str>) -> Option<String> {
    let lower = message.to_lowercase();
    let is_overflow = lower.contains("context size") || lower.contains("maximum allowed");
    if !is_overflow || lower.contains("maximum context length is") {
        return None;
    }
    Some(match model.and_then(context_window) {
        Some(n) => format!(
            "This model's maximum context length is {n} tokens. However, your messages resulted in \
             more tokens than that. Please reduce the length of the messages. (fm serve: {message})"
        ),
        None => format!(
            "This model's maximum context length was exceeded. Please reduce the length of the \
             messages. (fm serve: {message})"
        ),
    })
}

/// Rewrite an `{"error": {...}}` body if it reports a context overflow; `None` if it doesn't.
pub fn rewrite_overflow_error(body: &[u8], model: Option<&str>) -> Option<Vec<u8>> {
    let mut doc: Value = serde_json::from_slice(body).ok()?;
    let err = doc.get_mut("error")?.as_object_mut()?;
    let message = overflow_message(err.get("message")?.as_str()?, model)?;
    err.insert("message".into(), Value::from(message));
    err.insert("type".into(), Value::from("invalid_request_error"));
    err.insert("code".into(), Value::from("context_length_exceeded"));
    err.insert("param".into(), Value::from("messages"));
    Some(serde_json::to_vec(&doc).expect("serializing a JSON value can't fail"))
}

/// Rewrite complete SSE lines (each ending in `\n`): a `data:` line carrying a context-overflow
/// error gets the OpenAI-style error; everything else is copied. Returns the new bytes and
/// whether anything changed.
pub fn rewrite_sse_lines(lines: &[u8], model: Option<&str>) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(lines.len());
    let mut changed = false;
    for line in lines.split_inclusive(|&b| b == b'\n') {
        let replaced = line
            .strip_prefix(b"data:")
            .filter(|rest| rest.windows(7).any(|w| w == b"\"error\""))
            .and_then(|rest| rewrite_overflow_error(rest.trim_ascii(), model));
        match replaced {
            Some(body) => {
                out.extend_from_slice(b"data: ");
                out.extend_from_slice(&body);
                out.push(b'\n');
                changed = true;
            }
            None => out.extend_from_slice(line),
        }
    }
    (out, changed)
}

/// Collapse JSON Schema unions under `node`, recording each change in `notes`.
pub fn flatten(node: Value, path: &str, notes: &mut Vec<String>) -> Value {
    let mut obj = match node {
        Value::Array(items) => {
            return Value::Array(
                items
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| flatten(x, &format!("{path}[{i}]"), notes))
                    .collect(),
            );
        }
        Value::Object(obj) => obj,
        other => return other,
    };

    for key in ["anyOf", "oneOf"] {
        let all = match obj.shift_remove(key) {
            Some(Value::Array(all)) => all,
            Some(other) => {
                // Not a list: leave it alone.
                obj.insert(key.into(), other);
                continue;
            }
            None => continue,
        };
        let non_null: Vec<&Value> = all.iter().filter(|b| type_of(b) != Some("null")).collect();
        let branches: Vec<&Value> = if non_null.is_empty() {
            all.iter().collect()
        } else {
            non_null
        };
        let Some(first) = branches.first() else {
            notes.push(format!("{path}.{key}: empty, removed"));
            continue;
        };
        let pick = first.as_object().cloned().unwrap_or_default();
        let dropped: Vec<String> = branches[1..]
            .iter()
            .filter(|b| b.is_object())
            .map(|b| b.get("type").map_or("?".into(), repr_bare))
            .collect();
        notes.push(format!(
            "{path}.{key}: kept {}, dropped {}",
            pick.get("type").map_or("?".into(), repr_bare),
            dropped.join(",")
        ));
        let mut merged = pick.clone();
        for (k, v) in obj {
            if KEEP_FROM_PARENT.contains(&k.as_str()) || !pick.contains_key(&k) {
                merged.insert(k, v);
            }
        }
        obj = merged;
    }

    if let Some(Value::Array(types)) = obj.get("type") {
        let non_null: Vec<Value> = types
            .iter()
            .filter(|t| t.as_str() != Some("null"))
            .cloned()
            .collect();
        let kept = if non_null.is_empty() {
            types.clone()
        } else {
            non_null
        };
        if kept.len() > 1 {
            let rest: Vec<String> = kept[1..].iter().map(repr_bare).collect();
            notes.push(format!(
                "{path}.type: kept {}, dropped {}",
                repr_bare(&kept[0]),
                rest.join(",")
            ));
        }
        match kept.into_iter().next() {
            Some(t) => {
                obj.insert("type".into(), t);
            }
            None => {
                obj.shift_remove("type");
            }
        }
    }

    Value::Object(
        obj.into_iter()
            .map(|(k, v)| {
                let v = if DATA_KEYS.contains(&k.as_str()) {
                    v
                } else {
                    flatten(v, &format!("{path}.{k}"), notes)
                };
                (k, v)
            })
            .collect(),
    )
}

fn type_of(v: &Value) -> Option<&str> {
    v.get("type").and_then(Value::as_str)
}

/// Strings bare, everything else as JSON.
fn repr_bare(v: &Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_owned)
}

/// Strings in single quotes, everything else as JSON.
fn repr(v: &Value) -> String {
    v.as_str()
        .map_or_else(|| v.to_string(), |s| format!("'{s}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FM_OVERFLOW: &str = r#"{"error":{"message":"The session's transcript exceeded the model's context size.","type":"server_error","code":"500"}}"#;

    #[test]
    fn overflow_error_becomes_context_length_exceeded() {
        let out: Value = serde_json::from_slice(
            &rewrite_overflow_error(FM_OVERFLOW.as_bytes(), Some("pcc")).unwrap(),
        )
        .unwrap();
        assert_eq!(out["error"]["code"], "context_length_exceeded");
        assert_eq!(out["error"]["type"], "invalid_request_error");
        let msg = out["error"]["message"].as_str().unwrap();
        assert!(
            msg.starts_with("This model's maximum context length is 32768 tokens."),
            "{msg}"
        );
        // Already rewritten, and unrelated errors, are left alone.
        assert!(rewrite_overflow_error(msg.as_bytes(), Some("pcc")).is_none());
        let other = r#"{"error":{"message":"Invalid request: bad format","code":"400"}}"#;
        assert!(rewrite_overflow_error(other.as_bytes(), Some("pcc")).is_none());
    }

    #[test]
    fn overflow_in_sse_stream_is_rewritten_in_place() {
        let stream = format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}}}}]}}\n\nevent: error\ndata: {FM_OVERFLOW}\n\n"
        );
        let (out, changed) = rewrite_sse_lines(stream.as_bytes(), Some("system"));
        let out = String::from_utf8(out).unwrap();
        assert!(changed);
        assert!(out.starts_with("data: {\"choices\""));
        assert!(out.contains("event: error\ndata: {"));
        assert!(out.contains("maximum context length is 4096 tokens"));
        assert!(out.ends_with("\n\n"));
        let plain = "data: {\"choices\":[]}\n\n";
        assert_eq!(
            rewrite_sse_lines(plain.as_bytes(), Some("pcc")),
            (plain.as_bytes().to_vec(), false)
        );
    }

    fn run(v: Value) -> (Value, Rewrite) {
        let r = rewrite(&serde_json::to_vec(&v).unwrap());
        (serde_json::from_slice(&r.body).unwrap(), r)
    }

    #[test]
    fn any_of_keeps_first_non_null_branch_and_parent_description() {
        let s = json!({"description": "d", "anyOf": [{"type": "null"}, {"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]});
        let mut notes = vec![];
        assert_eq!(
            flatten(s, "t", &mut notes),
            json!({"type": "boolean", "description": "d"})
        );
        assert_eq!(notes, ["t.anyOf: kept boolean, dropped array"]);
    }

    #[test]
    fn one_of_and_type_arrays_are_collapsed_recursively() {
        let s = json!({"type": "object", "properties": {
            "a": {"type": ["string", "null"]},
            "b": {"type": ["integer", "string"]},
            "c": {"oneOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "integer"}]},
            "d": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}
        }, "required": ["a"]});
        let mut notes = vec![];
        assert_eq!(
            flatten(s, "t", &mut notes),
            json!({"type": "object", "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "string", "enum": ["x", "y"]},
                "d": {"type": "array", "items": {"type": "string"}}
            }, "required": ["a"]})
        );
        assert_eq!(notes.len(), 3, "{notes:?}"); // b, c, d; a only drops null
    }

    #[test]
    fn data_keys_are_not_rewritten() {
        let s = json!({"type": "object", "default": {"anyOf": 1}, "enum": [{"type": ["a", "b"]}]});
        let mut notes = vec![];
        assert_eq!(flatten(s.clone(), "t", &mut notes), s);
        assert!(notes.is_empty());
    }

    #[test]
    fn empty_union_is_removed() {
        let mut notes = vec![];
        assert_eq!(
            flatten(json!({"anyOf": [], "description": "d"}), "t", &mut notes),
            json!({"description": "d"})
        );
    }

    #[test]
    fn developer_role_becomes_system() {
        let (v, r) = run(
            json!({"model": "system", "messages": [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}]}),
        );
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(r.notes, ["developer role -> system"]);
        assert_eq!(
            r.summary.unwrap(),
            "model=system stream=None msgs=2 last_roles=['system', 'user'] tools=0 rewrites=1"
        );
    }

    #[test]
    fn reasoning_effort_rules() {
        let effort = |model: &str, e: Value| {
            run(json!({"model": model, "reasoning_effort": e}))
                .0
                .get("reasoning_effort")
                .cloned()
        };
        assert_eq!(effort("pcc", json!("minimal")), Some(json!("low")));
        assert_eq!(effort("pcc", json!("high")), Some(json!("high")));
        assert_eq!(effort("pcc", json!("none")), None);
        assert_eq!(effort("pcc", json!("xhigh")), None);
        assert_eq!(effort("system", json!("low")), None);
        assert_eq!(effort("system", Value::Null), Some(Value::Null));
    }

    #[test]
    fn tool_parameters_are_flattened() {
        let (v, r) = run(
            json!({"model": "pcc", "stream": true, "tools": [{"type": "function", "function": {"name": "terminal",
            "parameters": {"type": "object", "properties": {"notify": {"anyOf": [{"type": "boolean"}, {"type": "array"}]}}}}}]}),
        );
        assert_eq!(
            v["tools"][0]["function"]["parameters"]["properties"]["notify"],
            json!({"type": "boolean"})
        );
        assert_eq!(
            r.notes,
            ["terminal.properties.notify.anyOf: kept boolean, dropped array"]
        );
        assert!(
            r.summary
                .unwrap()
                .contains("stream=True msgs=0 last_roles=[] tools=1 rewrites=1")
        );
    }

    #[test]
    fn non_json_and_non_object_pass_through() {
        assert_eq!(rewrite(b"not json").body, b"not json");
        assert_eq!(rewrite(b"[1,2]").body, b"[1,2]");
        assert!(rewrite(b"[1,2]").summary.is_none());
    }
}
