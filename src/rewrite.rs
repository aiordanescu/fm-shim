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

use serde_json::{Map, Value};

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
}

pub fn rewrite(body: &[u8]) -> Rewrite {
    let mut doc: Map<String, Value> = match serde_json::from_slice(body) {
        Ok(Value::Object(m)) => m,
        _ => {
            return Rewrite {
                body: body.to_vec(),
                summary: None,
                notes: Vec::new(),
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
    }
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
