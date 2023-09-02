
import torch
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Softmax Benchmark')

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--input-dim', type=int, default=1024)
    parser.add_argument('--num-reps', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    batch = torch.rand(args.batch_size, args.input_dim, device=args.device)

    softmax = torch.nn.Softmax(dim=1).to(args.device)
    softmax.eval()

    start = time.perf_counter()
    for _ in range(args.num_reps):
        softmax(batch)
    end = time.perf_counter()
    elapsed = (end - start) / args.num_reps
    elapsed_ms = elapsed * 1000

    print(f"Softmax: hidden: {args.input_dim}, batch: {args.batch_size}")
    print(f"Torch-{args.device}: {elapsed_ms:.4f} ms")



if __name__ == "__main__":
    main()