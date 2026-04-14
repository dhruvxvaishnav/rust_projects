use std::net::TcpListener;

fn main(){
    let listener = TcpListener::bind("127.0.0.1:6379").expect
    ("Failed to bind to port 6379");
    println!("Mini Redis Listening on port 6379...");

    for stream in listener.incoming(){
        match stream{
            Ok(stream) => {
                println!("New Client {}", stream.peer_addr().
                unwrap());
            }
            Err(e) => {
                eprintln!("Connection Failed: {}", e)
            }
        }
    }
}