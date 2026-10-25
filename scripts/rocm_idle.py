import os, time, torch
def cpu_s():
    with open("/proc/self/stat") as f: st = f.read().split()
    return (int(st[13]) + int(st[14])) / os.sysconf("SC_CLK_TCK")
x = torch.randn(64, 64, device="cuda"); y = x @ x; torch.cuda.synchronize()
a = cpu_s(); time.sleep(6); b = cpu_s()
print(f"{os.environ.get('TAG','default')}: idle {((b - a) / 6 * 100):.0f}% of a core, threads {len(os.listdir('/proc/self/task'))}")
