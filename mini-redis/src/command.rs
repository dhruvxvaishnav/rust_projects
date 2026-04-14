use crate::store::{Store, Entry};

pub fn handle_command(cmd: &str, store: &Store) -> String {
    let parts: Vec<&str> = cmd.trim().splitn(4, ' ').collect();
    let mut db = store.lock().unwrap();

    match parts[0].to_uppercase().as_str() {
        "PING" => "+PONG\r\n".to_string(),
        "SET" if parts.len() >= 3 => {
            let entry = if parts.len() == 5 && parts[3].to_uppercase() == "EX" {
                let secs: u64 = parts[4].parse().unwrap_or(0);
                Entry::with_expiry(parts[2].to_string(), secs)
            } else {
                Entry::new(parts[2].to_string())
            };
            db.insert(parts[1].to_string(), entry);
            "+OK\r\n".to_string()
        }
        "GET" if parts.len() == 2 => {
            match db.get(parts[1]) {
                Some(entry) if !entry.is_expired() => {
                    format!("${}\r\n{}\r\n", entry.value.len(), entry.value)
                }
                Some(_) => { db.remove(parts[1]); "$-1\r\n".to_string() }
                None => "$-1\r\n".to_string(),
            }
        }
        "DEL" if parts.len() == 2 => {
            let removed = db.remove(parts[1]).is_some();
            format!(":{}\r\n", if removed { 1 } else { 0 })
        }
        "EXISTS" if parts.len() == 2 => {
            let exists = db.get(parts[1]).map_or(false, |e| !e.is_expired());
            format!(":{}\r\n", if exists { 1 } else { 0 })
        }
        "EXPIRE" if parts.len() == 3 => {
            let secs: u64 = parts[2].parse().unwrap_or(0);
            match db.get_mut(parts[1]) {
                Some(entry) => {
                    entry.expires_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(secs));
                    ":1\r\n".to_string()
                }
                None => ":0\r\n".to_string(),
            }
        }
        "TTL" if parts.len() == 2 => {
            match db.get(parts[1]) {
                Some(entry) => match entry.expires_at {
                    Some(t) => {
                        let remaining = t.saturating_duration_since(std::time::Instant::now());
                        format!(":{}\r\n", remaining.as_secs())
                    }
                    None => ":-1\r\n".to_string(),
                },
                None => ":-2\r\n".to_string(),
            }
        }
        "KEYS" => {
            let keys: Vec<&String> = db.iter()
                .filter(|(_, e)| !e.is_expired())
                .map(|(k, _)| k)
                .collect();
            let mut response = format!("*{}\r\n", keys.len());
            for key in keys {
                response.push_str(&format!("${}\r\n{}\r\n", key.len(), key));
            }
            response
        }
        "FLUSH" => {
            db.clear();
            "+OK\r\n".to_string()
        }
        "RENAME" if parts.len() == 3 => {
            match db.remove(parts[1]) {
                Some(entry) => { db.insert(parts[2].to_string(), entry); "+OK\r\n".to_string() }
                None => "-ERR no such key\r\n".to_string(),
            }
        }
        _ => "-ERR unknown command\r\n".to_string(),
    }
}