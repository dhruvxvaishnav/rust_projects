mod store;
mod command;

use std::io::{BufRead,BufReader,Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use crate::store::Store;

fn main(){
    let listener = TcpListener::bind("127.0.0.1:6379").expect
    ("Failed to bind to port 6379");
    println!("Mini Redis Listening on port 6379...");

    let store = store::new_store();

    for stream in listener.incoming(){
        match stream{
            Ok(stream) => {
                println!("New Client {}", stream.peer_addr().
                unwrap());
                let store = store.clone();
                thread::spawn(|| handle_client(stream, store));
            }
            Err(e) => {
                eprintln!("Connection Failed: {}", e)
            }
        }
    }
}

fn handle_client(mut stream: TcpStream, store:Store){
    let reader = BufReader::new(stream.try_clone().unwrap());

    for line in reader.lines(){
        match line {
            Ok(cmd) => {
                let response = command::handle_command(&cmd, &store);
                stream.write_all(response.as_bytes()).unwrap();
            }
        Err(_)=> break,
        }
    }
    println!("Client Disconnected")
} 