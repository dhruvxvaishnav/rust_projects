use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant,Duration};

pub struct Entry{
    pub value: String,
    pub expires_at:Option<Instant>,
}

impl Entry {
    pub fn new(value: String) -> Self {
        Entry { value, expires_at: None }
    }

    pub fn with_expiry(value:String, secs:u64)-> Self{
        Entry {
            value,
            expires_at:Some(Instant::now() + Duration::from_secs(secs)),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |t| Instant::now() > t)
    }
}


pub type Store = Arc<Mutex<HashMap<String, Entry>>>;

pub fn new_store() -> Store {
    Arc::new(Mutex::new(HashMap::new()))
}