import torch
def main():
    print(f"Cuda is available: {torch.cuda.is_available()}")
    
if __name__ == "__main__":
    main()