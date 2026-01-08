"""
Fully Fixed Advanced Neural ODE Engine - ALL dimension issues resolved
- Proper nested vmap patterns for attention
- Consistent batching throughout
- Fixed encoder/decoder dimension handling
- Validated all linear layer applications
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, value_and_grad
from jax.tree_util import tree_map
import equinox as eqx
import diffrax as dfx
import optax
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass
import chex
from functools import partial


@dataclass(frozen=True)
class ODEConfig:
    """Frozen configuration for JAX hashability"""
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    attention_heads: int = 8
    time_embedding_dim: int = 32
    solver: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    use_adjoint: bool = True
    dropout: float = 0.1
    sde_noise_scale: float = 0.01


class TemporalAttention(eqx.Module):
    """Multi-head attention with proper dimension handling"""
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    heads: int
    scale: float
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, *, key):
        # Validate dimensions
        if dim < heads:
            raise ValueError(f"dim ({dim}) must be >= heads ({heads})")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        keys = jr.split(key, 2)
        self.qkv_proj = eqx.nn.Linear(dim, dim * 3, key=keys[0])
        self.out_proj = eqx.nn.Linear(dim, dim, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout)
        
    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            output: [batch, seq_len, dim]
        """
        B, N, C = x.shape
        
        # FIXED: Proper nested vmap for [batch, seq] grid
        # Apply linear to each (batch, seq) position
        qkv = jax.vmap(jax.vmap(self.qkv_proj))(x)  # [B, N, 3*C]
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, N, 3, self.heads, C // self.heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        if not inference and key is not None:
            attn = self.dropout(attn, key=key)
        
        # Aggregate values
        x = jnp.einsum('bhqk,bhvd->bhqd', attn, v)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)
        
        # FIXED: Proper nested vmap for output projection
        x = jax.vmap(jax.vmap(self.out_proj))(x)
        
        return x


class TimeEmbedding(eqx.Module):
    """Sinusoidal time embeddings"""
    dim: int
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """t: [...] -> [..., dim]"""
        t = jnp.atleast_1d(t.squeeze())
        half_dim = self.dim // 2
        embeddings = jnp.log(10000.0) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings


class MLPBlock(eqx.Module):
    """MLP block - proper Equinox module"""
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    use_activation: bool
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, 
                 use_activation: bool = True, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
        self.norm = eqx.nn.LayerNorm(out_dim)
        self.dropout = eqx.nn.Dropout(dropout)
        self.use_activation = use_activation
        
    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        """Expects 1D input [in_dim] -> [out_dim]"""
        x = self.linear(x)
        x = self.norm(x)
        if self.use_activation:
            x = jax.nn.silu(x)
        if not inference and key is not None:
            x = self.dropout(x, key=key)
        return x


class LatentODEFunc(eqx.Module):
    """Neural ODE function: dz/dt = f(z, t, θ)"""
    time_embed: TimeEmbedding
    blocks: Tuple[MLPBlock, ...]
    temporal_attn: TemporalAttention
    latent_dim: int
    
    def __init__(self, config: ODEConfig, *, key):
        self.latent_dim = config.latent_dim
        self.time_embed = TimeEmbedding(config.time_embedding_dim)
        
        # Build MLP blocks
        keys = jr.split(key, config.num_layers + 1)
        dims = [config.latent_dim + config.time_embedding_dim] + \
               [config.hidden_dim] * config.num_layers + \
               [config.latent_dim]
        
        blocks = []
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            blocks.append(MLPBlock(
                dims[i], dims[i+1], 
                dropout=config.dropout,
                use_activation=not is_last,
                key=keys[i]
            ))
        
        self.blocks = tuple(blocks)
        self.temporal_attn = TemporalAttention(
            config.latent_dim, 
            config.attention_heads, 
            config.dropout,
            key=keys[-1]
        )
        
    def __call__(self, t: jnp.ndarray, z: jnp.ndarray, args=None) -> jnp.ndarray:
        """
        Compute dz/dt
        Args:
            t: Scalar time
            z: State [latent_dim] or [batch, latent_dim]
        Returns:
            dz/dt with same shape as z
        """
        single_input = z.ndim == 1
        if single_input:
            z = z[None, :]
        
        batch_size = z.shape[0]
        
        # Time embedding
        t_scalar = jnp.atleast_1d(t).squeeze()
        t_batch = jnp.broadcast_to(t_scalar, (batch_size,))
        t_embed = self.time_embed(t_batch)
        
        # Concatenate state with time
        zt = jnp.concatenate([z, t_embed], axis=-1)
        
        # Apply MLP blocks with proper vmap
        x = zt
        for block in self.blocks:
            x = jax.vmap(lambda xi: block(xi, inference=True))(x)
        
        dz = x
        
        # Apply temporal attention if batch > 1
        if batch_size > 1:
            z_attn = self.temporal_attn(z[None, :, :], inference=True)
            dz = dz + 0.1 * z_attn.squeeze(0)
        
        if single_input:
            dz = dz.squeeze(0)
        
        return dz


class ContinuousNormalizingFlow(eqx.Module):
    """CNF with Hutchinson trace estimator"""
    ode_func: LatentODEFunc
    rtol: float
    atol: float
    solver_name: str
    use_adjoint: bool
    
    def __init__(self, config: ODEConfig, *, key):
        self.ode_func = LatentODEFunc(config, key=key)
        self.rtol = config.rtol
        self.atol = config.atol
        self.solver_name = config.solver
        self.use_adjoint = config.use_adjoint
        
    def augmented_dynamics(self, t, y_aug, args):
        """Dynamics with log-det tracking"""
        z = y_aug[:-1]
        key = args
        
        dz = self.ode_func(t, z)
        
        # Hutchinson trace estimator
        eps = jr.normal(key, z.shape)
        _, dz_deps = jax.jvp(
            lambda z_: self.ode_func(t, z_),
            (z,),
            (eps,)
        )
        trace = jnp.sum(dz_deps * eps)
        
        return jnp.concatenate([dz, jnp.array([-trace])])
    
    def __call__(self, z0: jnp.ndarray, t0: float, t1: float, *, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward integration with log-det"""
        z0_aug = jnp.concatenate([z0, jnp.array([0.0])])
        
        term = dfx.ODETerm(self.augmented_dynamics)
        solver = {
            "dopri5": dfx.Dopri5(),
            "tsit5": dfx.Tsit5(),
            "kvaerno5": dfx.Kvaerno5(),
            "heun": dfx.Heun()
        }.get(self.solver_name, dfx.Dopri5())
        
        solution = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=z0_aug,
            args=key,
            stepsize_controller=dfx.PIDController(rtol=self.rtol, atol=self.atol),
            adjoint=dfx.RecursiveCheckpointAdjoint() if self.use_adjoint else dfx.DirectAdjoint(),
            max_steps=10000
        )
        
        z1_aug = solution.ys[-1]
        return z1_aug[:-1], z1_aug[-1]


class LatentODEEncoder(eqx.Module):
    """Encoder with GRU - FIXED dimension handling"""
    gru: eqx.nn.GRUCell
    attention: TemporalAttention
    fc_mu: eqx.nn.Linear
    fc_logvar: eqx.nn.Linear
    hidden_dim: int
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128, *, key):
        keys = jr.split(key, 4)
        self.hidden_dim = hidden_dim
        
        # GRU processes concatenated input+time
        self.gru = eqx.nn.GRUCell(input_dim + 1, hidden_dim, key=keys[0])
        
        # FIXED: Attention must match hidden_dim, with proper head count
        attn_heads = max(1, min(8, hidden_dim // 16))
        # Ensure divisibility
        if hidden_dim % attn_heads != 0:
            attn_heads = 1
        self.attention = TemporalAttention(hidden_dim, heads=attn_heads, key=keys[1])
        
        self.fc_mu = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[2])
        self.fc_logvar = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[3])
        
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, *, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: [batch, seq_len, input_dim]
            t: [batch, seq_len, 1]
        Returns:
            mu, logvar: [batch, latent_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Concatenate with time
        xt = jnp.concatenate([x, t], axis=-1)  # [batch, seq, input_dim+1]
        
        # Process each sequence with GRU
        def process_sequence(xt_seq):
            """xt_seq: [seq, input_dim+1] -> [seq, hidden_dim]"""
            def step_fn(h, xt_t):
                h_new = self.gru(xt_t, h)
                return h_new, h_new
            
            h0 = jnp.zeros(self.hidden_dim)
            _, h_seq = jax.lax.scan(step_fn, h0, xt_seq)
            return h_seq
        
        # FIXED: Proper vmap over batch dimension
        h_all = jax.vmap(process_sequence)(xt)  # [batch, seq, hidden_dim]
        
        # Apply attention
        h_attn = self.attention(h_all, key=key, inference=True)
        
        # Take final hidden state
        h_final = h_attn[:, -1, :]  # [batch, hidden_dim]
        
        # FIXED: Proper vmap for fc layers
        mu = jax.vmap(self.fc_mu)(h_final)
        logvar = jax.vmap(self.fc_logvar)(h_final)
        
        return mu, logvar


class LatentODEDecoder(eqx.Module):
    """Decoder - FIXED to handle batched inputs properly"""
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128, *, key):
        keys = jr.split(key, 3)
        self.layer1 = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.layer2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
        self.layer3 = eqx.nn.Linear(hidden_dim, output_dim, key=keys[2])
        
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Handles arbitrary batch dimensions via vmap
        z: [latent_dim] or [batch, latent_dim] or [batch, seq, latent_dim]
        """
        def decode_single(z_single):
            """Decode single latent vector"""
            x = jax.nn.silu(self.layer1(z_single))
            x = jax.nn.silu(self.layer2(x))
            x = self.layer3(x)
            return x
        
        # Handle different input shapes
        if z.ndim == 1:
            # Single vector
            return decode_single(z)
        elif z.ndim == 2:
            # [batch, latent_dim]
            return jax.vmap(decode_single)(z)
        elif z.ndim == 3:
            # [batch, seq, latent_dim]
            return jax.vmap(jax.vmap(decode_single))(z)
        else:
            raise ValueError(f"Unexpected input shape: {z.shape}")


class NeuralODEConsistencyPredictor(eqx.Module):
    """Main Neural ODE system"""
    encoder: LatentODEEncoder
    ode_func: LatentODEFunc
    cnf: ContinuousNormalizingFlow
    decoder: LatentODEDecoder
    config: ODEConfig
    
    def __init__(self, input_dim: int = 768, config: Optional[ODEConfig] = None, *, key):
        self.config = config or ODEConfig()
        keys = jr.split(key, 4)
        
        # FIXED: Proper encoder hidden dimension calculation
        encoder_hidden = max(32, self.config.latent_dim * 2)
        # Ensure divisibility for attention
        if encoder_hidden < 16:
            encoder_hidden = 16
        
        self.encoder = LatentODEEncoder(
            input_dim, 
            self.config.latent_dim,
            hidden_dim=encoder_hidden,
            key=keys[0]
        )
        self.ode_func = LatentODEFunc(self.config, key=keys[1])
        self.cnf = ContinuousNormalizingFlow(self.config, key=keys[2])
        self.decoder = LatentODEDecoder(
            self.config.latent_dim, 
            input_dim,
            hidden_dim=max(32, self.config.latent_dim),
            key=keys[3]
        )
    
    def reparameterize(self, mu: jnp.ndarray, logvar: jnp.ndarray, *, key) -> jnp.ndarray:
        """VAE reparameterization"""
        std = jnp.exp(0.5 * logvar)
        eps = jr.normal(key, mu.shape)
        return mu + eps * std
    
    def predict_with_uncertainty(
        self,
        x_history: jnp.ndarray,
        t_history: jnp.ndarray,
        t_target: jnp.ndarray,
        n_samples: int = 50,
        *,
        key
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict with uncertainty quantification"""
        keys = jr.split(key, n_samples + 1)
        
        # Encode
        mu, logvar = self.encoder(x_history, t_history, key=keys[0])
        
        t0 = float(t_history[0, -1, 0])
        
        # Sample trajectories
        def sample_trajectory(sample_key):
            z0 = self.reparameterize(mu, logvar, key=sample_key)
            
            term = dfx.ODETerm(self.ode_func)
            solver = dfx.Dopri5() if self.config.solver == "dopri5" else dfx.Tsit5()
            
            # Predict for each target time
            predictions = []
            for t_tgt in t_target[0]:
                solution = dfx.diffeqsolve(
                    term,
                    solver,
                    t0=t0,
                    t1=float(t_tgt),
                    dt0=0.01,
                    y0=z0[0],
                    stepsize_controller=dfx.PIDController(rtol=self.config.rtol, atol=self.config.atol),
                    max_steps=5000
                )
                z_pred = solution.ys[-1]
                # FIXED: Decoder handles single vector properly
                x_pred = self.decoder(z_pred)
                predictions.append(x_pred)
            
            return jnp.stack(predictions)
        
        # Vectorize sampling
        all_predictions = jax.vmap(sample_trajectory)(keys[1:n_samples+1])
        
        # Statistics
        pred_mean = jnp.mean(all_predictions, axis=0)
        pred_std = jnp.std(all_predictions, axis=0)
        anomaly_scores = jnp.mean(pred_std, axis=-1)
        
        return pred_mean, pred_std, anomaly_scores
    
    def compute_trajectory_loss(
        self,
        x_true: jnp.ndarray,
        x_pred: jnp.ndarray,
        mu: jnp.ndarray,
        logvar: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """ELBO-based loss"""
        recon_loss = jnp.mean((x_pred - x_true) ** 2)
        kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar)) / mu.shape[0]
        
        smoothness_loss = 0.0
        if x_pred.ndim > 1 and x_pred.shape[0] > 2:
            first_diff = jnp.diff(x_pred, axis=0)
            second_diff = jnp.diff(first_diff, axis=0)
            smoothness_loss = jnp.mean(second_diff ** 2)
        
        total_loss = recon_loss + 0.001 * kl_loss + 0.01 * smoothness_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'smoothness': smoothness_loss
        }


class StiffnessDetector:
    """Detects ODE stiffness"""
    
    @staticmethod
    def detect_stiffness(
        func: Callable,
        z: jnp.ndarray,
        t: float,
        threshold: float = 100.0
    ) -> Tuple[bool, float]:
        jacobian = jax.jacfwd(lambda z_: func(t, z_))(z)
        eigenvalues = jnp.linalg.eigvals(jacobian)
        eigenvalues_abs = jnp.abs(eigenvalues)
        
        max_eig = jnp.max(eigenvalues_abs)
        min_eig = jnp.min(eigenvalues_abs[eigenvalues_abs > 1e-10])
        ratio = max_eig / (min_eig + 1e-10)
        
        return ratio > threshold, float(ratio)
    
    @staticmethod
    def suggest_solver(is_stiff: bool) -> str:
        return "kvaerno5" if is_stiff else "dopri5"


# Training utilities
def create_optimizer(learning_rate: float = 1e-4):
    """Create Optax optimizer"""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=10000,
        end_value=1e-6
    )
    
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-5),
    )


@eqx.filter_jit
def train_step(
    model: NeuralODEConsistencyPredictor,
    opt_state,
    optimizer,
    x_batch: jnp.ndarray,
    t_batch: jnp.ndarray,
    x_target: jnp.ndarray,
    key
):
    """Single training step"""
    
    def loss_fn(model):
        keys = jr.split(key, 2)
        mu, logvar = model.encoder(x_batch, t_batch, key=keys[0])
        z0 = model.reparameterize(mu, logvar, key=keys[1])
        # FIXED: Decoder handles batched input
        x_pred = model.decoder(z0)
        losses = model.compute_trajectory_loss(x_target, x_pred, mu, logvar)
        return losses['total'], losses
    
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss, aux


# Flask API with thread-safe state
from flask import Flask, request, jsonify
import numpy as np
import threading

app = Flask(__name__)


class ModelState:
    """Thread-safe model state management"""
    def __init__(self):
        self.model = None
        self.config = None
        self.input_dim = None
        self.lock = threading.Lock()
        self.key_counter = 0
    
    def get_next_key(self):
        """Thread-safe key generation"""
        with self.lock:
            self.key_counter += 1
            return jr.PRNGKey(self.key_counter)
    
    def initialize_model(self, input_dim: int):
        """Thread-safe model initialization with smart scaling"""
        with self.lock:
            if self.model is None or self.input_dim != input_dim:
                print(f"\n{'='*60}")
                print(f"Initializing model with input_dim={input_dim}")
                self.input_dim = input_dim
                
                # FIXED: Smart dimension scaling
                latent_dim = max(16, min(128, input_dim * 8))
                attention_heads = min(8, latent_dim // 16)
                attention_heads = max(1, attention_heads)
                
                # Ensure divisibility
                latent_dim = attention_heads * (latent_dim // attention_heads)
                
                self.config = ODEConfig(
                    latent_dim=latent_dim,
                    hidden_dim=max(64, latent_dim * 2),
                    num_layers=3 if input_dim < 32 else 4,
                    attention_heads=attention_heads,
                    solver='dopri5',
                    sde_noise_scale=0.01
                )
                
                init_key = jr.PRNGKey(42)
                self.model = NeuralODEConsistencyPredictor(
                    input_dim=input_dim,
                    config=self.config,
                    key=init_key
                )
                
                param_count = sum(
                    x.size for x in jax.tree_util.tree_leaves(
                        eqx.filter(self.model, eqx.is_array)
                    )
                )
                print(f"Architecture: latent={latent_dim}, heads={attention_heads}")
                print(f"Total parameters: {param_count:,}")
                print(f"{'='*60}\n")


state = ModelState()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': state.model is not None,
        'backend': 'JAX/Equinox/Diffrax',
        'devices': str(jax.devices()),
        'input_dim': state.input_dim,
        'config': {
            'latent_dim': state.config.latent_dim if state.config else None,
            'attention_heads': state.config.attention_heads if state.config else None
        } if state.config else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict using JAX model"""
    try:
        data = request.json
        
        vectors = data['vectors']
        input_dim = len(vectors[0])
        
        # Initialize if needed
        state.initialize_model(input_dim)
        
        # Prepare data
        x_history = jnp.array(vectors, dtype=jnp.float32)[None, :, :]
        t_history = jnp.array(data['timestamps'], dtype=jnp.float32)[None, :, None]
        t_target = jnp.array([[data['target_time']]], dtype=jnp.float32)
        
        # Get thread-safe key
        key = state.get_next_key()
        
        # Predict
        pred_mean, pred_std, anomaly_scores = state.model.predict_with_uncertainty(
            x_history, t_history, t_target, n_samples=10, key=key
        )
        
        return jsonify({
            # FIXED: Access [0] for the first target time, not [0,0]
            'predicted_vector': pred_mean[0].tolist(),       # Returns full vector [dim]
            'uncertainty': float(pred_std[0].mean()),        # Mean uncertainty across dims
            'anomaly_score': float(anomaly_scores[0]),       # Scalar score for target
            'is_anomalous': float(anomaly_scores[0]) > 0.5,
            'backend': 'JAX/Equinox/Diffrax',
            'samples_used': 10,
            'input_dim': input_dim,
            'latent_dim': state.config.latent_dim
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400


@app.route('/stiffness', methods=['POST'])
def check_stiffness():
    """Check ODE stiffness"""
    try:
        data = request.json
        z = jnp.array(data['state'])
        t = float(data['time'])
        
        is_stiff, ratio = StiffnessDetector.detect_stiffness(
            state.model.ode_func, z, t
        )
        
        suggested_solver = StiffnessDetector.suggest_solver(is_stiff)
        
        return jsonify({
            'is_stiff': bool(is_stiff),
            'stiffness_ratio': float(ratio),
            'suggested_solver': suggested_solver,
            'current_solver': state.config.solver
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 70)
    print("Neural ODE Temporal Consistency Engine (JAX/Equinox/Diffrax)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()
    print("✓ All dimension handling issues fixed")
    print("✓ Proper nested vmap patterns")
    print("✓ Smart architecture scaling")
    print("✓ Thread-safe state management")
    print()
    print("Model initializes on first request")
    print("Service ready on http://0.0.0.0:5000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)