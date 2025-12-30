import time
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

# Define a simple vector field for benchmarking
class BenchmarkField(eqx.Module):
    scale: float = 1.0

    def __call__(self, t, y, args):
        return -self.scale * y

def run_benchmark(solver_name, solver, y0, t1, dt0, steps=100):
    term = diffrax.ODETerm(BenchmarkField())
    
    # Warmup (JIT Compilation)
    diffrax.diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt0, y0=y0)
    
    start = time.time()
    for _ in range(steps):
        diffrax.diffeqsolve(term, solver, t0=0, t1=t1, dt0=dt0, y0=y0)
    end = time.time()
    
    avg_time = (end - start) / steps * 1000
    print(f"{solver_name}: {avg_time:.4f} ms per solve")

if __name__ == "__main__":
    print("Benchmarking JAX ODE Solvers...")
    
    y0 = jnp.ones((1024,)) # 1024-dim vector
    t1 = 1.0
    dt0 = 0.1
    
    solvers = {
        "Euler": diffrax.Euler(),
        "Dopri5": diffrax.Dopri5(),
        "Tsit5": diffrax.Tsit5(),
        "Heun": diffrax.Heun(),
    }
    
    for name, solver in solvers.items():
        run_benchmark(name, solver, y0, t1, dt0)