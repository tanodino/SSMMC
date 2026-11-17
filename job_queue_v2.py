"""
job_queue_v2.py -- job_queue.py plus two options to share the workstation more gently:
  --adopt GPU:PID   count an already-running process (e.g. left by a previous queue) on GPU
                    as one of our jobs while it is alive (no new launch takes its slot);
  --threads N       OMP/MKL threads (and torch intra-op threads) for each launched job.

job_queue.py -- run a list of GPU jobs on a FIXED set of GPUs, several jobs per
GPU when memory allows. Used to run the baselines on the shared workstation
without ever using more than the allowed GPUs.

    python job_queue.py JOBS_FILE --gpus 0 3 --max_per_gpu 3 --min_free_gb 12 \
        --gate 3:/path/status.txt:SUNRGBD --gate 0:/path/status.txt:TRISTAR \
        --log_dir logs/baselines

JOBS_FILE: one shell command per line (no CUDA_VISIBLE_DEVICES; it is added),
           blank lines and lines starting with # are ignored.
A job is launched on a GPU only if
  * the GPU's gate (if any) is open: the gate file contains the given text
    (e.g. "SUNRGBD" once the pretraining on that GPU wrote "SUNRGBD exit=0"),
  * fewer than --max_per_gpu of our jobs run on it,
  * it has at least --min_free_gb of free memory (nvidia-smi),
  * at least --settle_s seconds passed since our last launch on that GPU.
State lives in <log_dir>: done.txt / failed.txt (job line numbers), one log per
job. Re-running the same command resumes: done jobs are skipped; a failed job
is retried up to --max_attempts times.
"""

import argparse
import os
import subprocess
import time


def free_mem_gb(gpu):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                                   "-i", str(gpu)], text=True)
    return float(out.strip()) / 1024


def read_ids(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [int(l.split()[0]) for l in fh if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs_file")
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--max_per_gpu", type=int, default=3)
    p.add_argument("--min_free_gb", type=float, default=12.0)
    p.add_argument("--settle_s", type=int, default=180)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--gate", action="append", default=[], help="GPU:FILE:TEXT")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--adopt", action="append", default=[], help="GPU:PID of an external job to count")
    p.add_argument("--threads", type=int, default=0, help="OMP/MKL threads per job (0 = unchanged)")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    done_path = os.path.join(args.log_dir, "done.txt")
    failed_path = os.path.join(args.log_dir, "failed.txt")
    gates = {}
    for g in args.gate:
        gpu, path, text = g.split(":", 2)
        gates[int(gpu)] = (path, text)

    with open(args.jobs_file) as fh:
        jobs = [(i, l.strip()) for i, l in enumerate(fh, 1) if l.strip() and not l.lstrip().startswith("#")]
    done = set(read_ids(done_path))
    attempts = {}
    for i in read_ids(failed_path):
        attempts[i] = attempts.get(i, 0) + 1
    pending = [(i, c) for i, c in jobs if i not in done and attempts.get(i, 0) < args.max_attempts]
    running = {}            # job id -> (Popen, gpu, cmd, log file handle)
    last_launch = {g: 0.0 for g in args.gpus}
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(pending)} to run on GPUs {args.gpus}", flush=True)

    adopted = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in args.adopt]

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def gate_open(gpu):
        if gpu not in gates:
            return True
        path, text = gates[gpu]
        return os.path.exists(path) and text in open(path).read()

    while pending or running or any(alive(p) for _, p in adopted):
        # collect finished jobs
        for jid in list(running):
            proc, gpu, cmd, fh = running[jid]
            if proc.poll() is not None:
                fh.close()
                ok = proc.returncode == 0
                with open(done_path if ok else failed_path, "a") as f:
                    f.write(f"{jid} gpu={gpu} exit={proc.returncode} {cmd}\n")
                print(f"{time.strftime('%H:%M:%S')} {'DONE' if ok else 'FAIL'} job {jid} (gpu {gpu}, exit "
                      f"{proc.returncode}) | running {len(running) - 1} pending {len(pending)}", flush=True)
                if not ok:
                    attempts[jid] = attempts.get(jid, 0) + 1
                    if attempts[jid] < args.max_attempts:
                        pending.append((jid, cmd))        # retry later
                del running[jid]
        # launch new jobs
        for gpu in args.gpus:
            if not pending:
                break
            n_on_gpu = sum(1 for v in running.values() if v[1] == gpu) + sum(1 for g, pid in adopted if g == gpu and alive(pid))
            if (not gate_open(gpu) or n_on_gpu >= args.max_per_gpu
                    or time.time() - last_launch[gpu] < args.settle_s or free_mem_gb(gpu) < args.min_free_gb):
                continue
            jid, cmd = pending.pop(0)
            fh = open(os.path.join(args.log_dir, f"job{jid:04d}.log"), "w")
            fh.write(f"# gpu {gpu} | {cmd}\n")
            fh.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            if args.threads > 0:
                env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            running[jid] = (subprocess.Popen(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env),
                            gpu, cmd, fh)
            last_launch[gpu] = time.time()
            print(f"{time.strftime('%H:%M:%S')} START job {jid} on gpu {gpu} ({n_on_gpu + 1} there) | "
                  f"pending {len(pending)}: {cmd}", flush=True)
        time.sleep(20)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
