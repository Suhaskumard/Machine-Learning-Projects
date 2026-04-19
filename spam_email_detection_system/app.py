from predictor import predict_email

def main():
    print("📧 Spam Email Detector")
    print("----------------------")
    
    while True:
        email = input("\nEnter email text (or 'exit'): ")
        
        if email.lower() == "exit":
            print("👋 Exiting...")
            break
        
        result = predict_email(email)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
