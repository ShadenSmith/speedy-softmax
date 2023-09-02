import torch
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Softmax Benchmark")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--num-reps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    batch = torch.rand(args.batch_size, args.input_dim, device=args.device)

    softmax = torch.nn.Softmax(dim=1).to(args.device)
    softmax.eval()

    start = time.perf_counter()
    for _ in range(args.num_reps):
        out = softmax(batch)
    _ = out.sum().item()  # force device sync
    end = time.perf_counter()
    elapsed = (end - start) / args.num_reps
    elapsed_ms = elapsed * 1000

    batch_traffic = args.batch_size * args.input_dim * batch.element_size() * 2
    bw_gbps = batch_traffic / elapsed / (1024**3)

    print(f"Softmax batch_size: {args.batch_size} input_dim: {args.input_dim}")
    print(f"Torch-{args.device}: {elapsed_ms:.4f} ms  [{bw_gbps:.2f} GB/s]")


if __name__ == "__main__":
    main()
