# Mini Redis Clone — Learning Project

A Redis-like in-memory key-value store built in Rust. This was a learning project to understand systems programming concepts in Rust.

---

## What I Built

A TCP server that listens on port `6379` (same as real Redis) and handles commands like `SET`, `GET`, `DEL`, `EXPIRE`, `TTL`, `KEYS`, `FLUSH`, and `RENAME`. Data persists to disk on shutdown and reloads on startup.

---

## What I Learned

### 1. Ownership & Borrowing

Rust's core concept — every value has one owner. When passing values into threads, you can't just share them freely. I ran into "use of moved value" errors when trying to share the store across multiple threads in a loop.

### 2. `Arc<Mutex<>>` — Shared State Across Threads

- `Arc` (Atomic Reference Counter) lets multiple threads share the same data by cloning a pointer, not the actual data
- `Mutex` ensures only one thread accesses the HashMap at a time
- Every time a new client connected, I cloned the `Arc` — all clones pointed to the same HashMap in memory

```rust
let store: Arc<Mutex<HashMap<String, Entry>>> = Arc::new(Mutex::new(HashMap::new()));
let store_clone = store.clone(); // clones the pointer, not the data
```

### 3. TCP Networking with `std::net`

- `TcpListener::bind()` to start a server
- `listener.incoming()` to accept connections in a loop
- `BufReader` to read lines from a TCP stream
- `stream.write_all()` to send responses back

### 4. Spawning Threads

Each client connection runs in its own thread using `thread::spawn`. This means multiple clients can connect simultaneously without blocking each other.

### 5. Pattern Matching

Used extensively for command parsing — matching on command names with guards like `if parts.len() == 2` to ensure the right number of arguments:

```rust
match parts[0].to_uppercase().as_str() {
    "GET" if parts.len() == 2 => { ... }
    "SET" if parts.len() >= 3 => { ... }
    _ => "-ERR unknown command\r\n".to_string(),
}
```

### 6. Structs & Methods

Created an `Entry` struct to hold both a value and an optional expiry time. Learned how `impl` blocks work to attach methods like `is_expired()` to a struct.

### 7. `Option<T>` and `map_or`

Used `Option` to represent keys that may or may not have an expiry. `.map_or(false, |t| ...)` is a clean way to handle the None case with a default value.

### 8. File I/O & Persistence

Used `std::fs::write` and `std::fs::read_to_string` to save and load data. On Ctrl+C, the server saves all keys to `dump.rdb` and reloads them on next startup.

### 9. Modules (`mod`)

Split code across multiple files:

- `store.rs` — data structures
- `command.rs` — command handling logic
- `persistence.rs` — save/load to disk
- `main.rs` — networking and server loop

Learned that `mod filename;` in `main.rs` tells Rust to include that file, and `pub` makes items accessible from other modules.

### 10. RESP Protocol

Redis uses the RESP (Redis Serialization Protocol) for communication:

- `+OK` — simple string
- `$4\r\nrust` — bulk string (4 = length of value)
- `$-1` — null (key not found)
- `:1` — integer
- `*2` — array of 2 items
- `-ERR message` — error

---

## Commands Supported

| Command  | Example               | Description                  |
| -------- | --------------------- | ---------------------------- |
| `PING`   | `PING`                | Health check, returns PONG   |
| `SET`    | `SET name rust`       | Store a key-value pair       |
| `SET EX` | `SET name rust EX 10` | Store with expiry in seconds |
| `GET`    | `GET name`            | Retrieve a value             |
| `DEL`    | `DEL name`            | Delete a key                 |
| `EXISTS` | `EXISTS name`         | Check if key exists          |
| `EXPIRE` | `EXPIRE name 10`      | Set expiry on existing key   |
| `TTL`    | `TTL name`            | Check remaining time to live |
| `KEYS`   | `KEYS`                | List all keys                |
| `RENAME` | `RENAME old new`      | Rename a key                 |
| `FLUSH`  | `FLUSH`               | Delete all keys              |

---

## How to Run

```bash
cargo run
```

Then in another terminal:

```bash
telnet 127.0.0.1 6379
```

Press `Ctrl+C` to stop the server — data will be saved to `dump.rdb` automatically.

---

## Project Structure

```text
src/
  main.rs         # TCP server, thread spawning
  store.rs        # Entry struct, shared store type
  command.rs      # Command parsing and execution
  persistence.rs  # Save/load to disk
Cargo.toml
dump.rdb          # Auto-generated on shutdown
```
