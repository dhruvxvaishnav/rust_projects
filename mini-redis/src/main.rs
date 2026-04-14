use std::io::{BufRead,BufReader,Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn main(){
    let listener = TcpListener::bind("127.0.0.1:6379").expect
    ("Failed to bind to port 6379");
    println!("Mini Redis Listening on port 6379...");

    for stream in listener.incoming(){
        match stream{
            Ok(stream) => {
                println!("New Client {}", stream.peer_addr().
                unwrap());
                thread::spawn(|| handle_client(stream));
            }
            Err(e) => {
                eprintln!("Connection Failed: {}", e)
            }
        }
    }
}

fn handle_client(mut stream: std::net::TcpStream){
    let reader = BufReader::new(stream.try_clone().unwrap());

    for line in reader.lines(){
        match line {
            Ok(cmd) => {
                println!("Received: {}", cmd);
                stream.write_all(b"+OK\r\n").unwrap();
            }
        Err(_)=> break,
        }
    }
    println!("Client Disconnected")
} 