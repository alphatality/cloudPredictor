import numpy as np
import pandas as pd


incident_rate = 0.035  # ~3.5% base incidence rate


def generate_dataset(n_steps: int = 14400) -> pd.DataFrame:
    """ 
    Three incident types at random positions:
      - cpu_spike:    abrupt +40 pp CPU → latency 2 min later
      - memory_leak:  slow linear ramp on memory over 15-25 min
      - thundering:   coordinated spike on all three channels
    """

    t = np.arange(n_steps)
 
    # Diurnal envelope (peak at t mod 1440 = 540, 9:00)
    diurnal = 8 * np.sin(2 * np.pi * (t % 1440 - 540) / 1440)
 
    cpu     = np.clip(40 + diurnal + 4 * np.random.randn(n_steps), 0, 100)
    memory  = np.clip(52 + 2 * np.random.randn(n_steps), 0, 100)
    latency = np.clip(55 + 0.3 * cpu + 3 * np.random.randn(n_steps), 0, 600)
 
    incident = np.zeros(n_steps, dtype=np.int8)
 
    i = 0
    while i < n_steps - 30:
        if np.random.rand() < incident_rate:               # ~3.5% base incidence rate
            kind = np.random.choice(["cpu_spike", "memory_leak", "thundering"],
                                     p=[0.5, 0.35, 0.15])
            dur  = int(np.random.uniform(8, 28))
            end  = min(i + dur, n_steps)
 
            if kind == "cpu_spike":
                cpu[i:end] += np.random.uniform(35, 55)
                lag = min(3, end - i - 1)
                latency[i + lag : end] += np.random.uniform(60, 120)
 
            elif kind == "memory_leak":
                ramp = np.linspace(0, np.random.uniform(25, 40), end - i)
                memory[i:end] += ramp
 
            else: 
                cpu[i:end]     += np.random.uniform(30, 50)
                memory[i:end]  += np.random.uniform(20, 35)
                latency[i:end] += np.random.uniform(80, 150)
 
            incident[i:end] = 1
            i = end + int(np.random.uniform(15, 60))  
        else:
            i += 1
 
    cpu     = np.clip(cpu, 0, 100)
    memory  = np.clip(memory, 0, 100)
    latency = np.clip(latency, 0, 600)
 
    return pd.DataFrame({
        "cpu":      cpu,
        "memory":   memory,
        "latency":  latency,
        "incident": incident,
        "cpu_latency": cpu/latency
    })
with open("data/synthetic_cloud_data.csv", "w") as f:
    f.write(generate_dataset().to_csv(index=True))