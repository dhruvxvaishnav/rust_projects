use crate::store::{Entry, Store};
use std::fs;

const SAVE_FILE: &str = "dump.rdb";

pub fn save(store: &Store) {
    let db = store.lock().unwrap();
    let mut lines = String::new();
    for (key, entry) in db.iter() {
        if !entry.is_expired() {
            lines.push_str(&format!("{}\n{}\n", key, entry.value));
        }
    }
    fs::write(SAVE_FILE, lines).expect("Failed to save");
    println!("Data saved to {}", SAVE_FILE);
}

pub fn load(store: &Store) {
    let Ok(contents) = fs::read_to_string(SAVE_FILE) else {
        return;
    };
    let mut db = store.lock().unwrap();
    let mut lines = contents.lines();
    while let (Some(key), Some(val)) = (lines.next(), lines.next()) {
        db.insert(key.to_string(), Entry::new(val.to_string()));
    }
    println!("Data loaded from {}", SAVE_FILE);
}
