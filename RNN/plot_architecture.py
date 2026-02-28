import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.set_xlim(-0.5, 21)
ax.set_ylim(-1.5, 5.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(10.25, 5.2, 'Attention-Based LSTM Architecture', fontsize=18, fontweight='bold',
        ha='center', va='center', fontfamily='serif')
ax.text(10.25, 4.7, 'Multi-Step Ahead UAV Attitude Prediction', fontsize=12,
        ha='center', va='center', fontfamily='serif', color='#555555')

# ── Block definitions ──────────────────────────────────────────────
# (x, y, width, height, facecolor, edgecolor)
blocks = [
    # Input
    {'x': 0, 'y': 0.5, 'w': 3.2, 'h': 3.5,
     'fc': '#D6EAF8', 'ec': '#2980B9',
     'title': 'Input',
     'lines': [
         'X ∈ ℝ [B × 10 × 9]',
         '',
         '10 timesteps',
         '9 features each:',
         '',
         '• (φ, θ, ψ)  angles',
         '• (ωφ, ωθ, ωψ)  gyro',
         '• (τφ, τθ, τψ)  torques',
     ]},
    # LSTM
    {'x': 4.2, 'y': 0.5, 'w': 3.2, 'h': 3.5,
     'fc': '#FDEBD0', 'ec': '#E67E22',
     'title': 'LSTM Layer',
     'lines': [
         'input_size = 9',
         'hidden_size = 128',
         'batch_first = true',
         '',
         'Processes full',
         'sequence of K=10',
         'timesteps',
         '',
         'Output: H ∈ [B,10,128]',
     ]},
    # Attention
    {'x': 8.4, 'y': 0.5, 'w': 3.8, 'h': 3.5,
     'fc': '#E8DAEF', 'ec': '#8E44AD',
     'title': 'Attention Mechanism',
     'lines': [
         '1. Energy:',
         '   e = tanh(Wₐ·H + bₐ)',
         '2. Scores:',
         '   s = vᵀ·e + bᵥ',
         '3. Weights:',
         '   α = softmax(s)',
         '4. Context vector:',
         '   c = Σ αₜ · hₜ',
         'Output: c ∈ [B, 128]',
     ]},
    # FC
    {'x': 13.2, 'y': 0.8, 'w': 3.0, 'h': 2.9,
     'fc': '#FADBD8', 'ec': '#E74C3C',
     'title': 'Fully Connected',
     'lines': [
         'Linear Layer',
         '',
         'in_features = 128',
         'out_features = 30',
         '',
         'No activation',
         '(regression output)',
         '',
     ]},
    # Output
    {'x': 17.2, 'y': 0.8, 'w': 3.2, 'h': 2.9,
     'fc': '#D5F5E3', 'ec': '#27AE60',
     'title': 'Output',
     'lines': [
         'Reshape',
         '30 → [10 × 3]',
         '',
         'Predicted angles',
         'for N=10 future',
         'timesteps:',
         '',
         '(φ̂, θ̂, ψ̂) per step',
     ]},
]

# ── Draw blocks ────────────────────────────────────────────────────
for b in blocks:
    # Rounded rectangle
    fancy = FancyBboxPatch(
        (b['x'], b['y']), b['w'], b['h'],
        boxstyle="round,pad=0.15",
        facecolor=b['fc'], edgecolor=b['ec'], linewidth=2.0
    )
    ax.add_patch(fancy)

    # Title
    ax.text(b['x'] + b['w']/2, b['y'] + b['h'] - 0.35,
            b['title'], fontsize=11, fontweight='bold',
            ha='center', va='center', fontfamily='serif',
            color=b['ec'])

    # Separator line under title
    ax.plot([b['x'] + 0.2, b['x'] + b['w'] - 0.2],
            [b['y'] + b['h'] - 0.6, b['y'] + b['h'] - 0.6],
            color=b['ec'], linewidth=0.8, alpha=0.5)

    # Content lines
    start_y = b['y'] + b['h'] - 0.85
    for i, line in enumerate(b['lines']):
        ax.text(b['x'] + 0.25, start_y - i * 0.28,
                line, fontsize=8.5, ha='left', va='center',
                fontfamily='monospace', color='#2C3E50')

# ── Draw arrows between blocks ─────────────────────────────────────
arrow_style = "Simple,tail_width=4,head_width=12,head_length=8"
arrow_color = '#7F8C8D'

arrow_pairs = [
    (3.2, 4.2),      # Input → LSTM
    (7.4, 8.4),      # LSTM → Attention
    (12.2, 13.2),    # Attention → FC
    (16.2, 17.2),    # FC → Output
]

for x_start, x_end in arrow_pairs:
    y_mid = 2.25
    arrow = FancyArrowPatch(
        (x_start + 0.1, y_mid), (x_end - 0.1, y_mid),
        arrowstyle=arrow_style,
        color=arrow_color, linewidth=1.5,
        mutation_scale=1
    )
    ax.add_patch(arrow)

# ── Tensor shape labels on arrows ──────────────────────────────────
shape_labels = [
    (3.7, '[B, 10, 9]'),
    (7.9, '[B, 10, 128]'),
    (12.7, '[B, 128]'),
    (16.7, '[B, 30]'),
]

for x_pos, label in shape_labels:
    ax.text(x_pos, 2.7, label, fontsize=7.5, ha='center', va='center',
            fontfamily='monospace', color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#BDC3C7', linewidth=0.5))

# ── Bottom annotation ──────────────────────────────────────────────
ax.text(10.25, -0.8,
        'Implementation: C++ with LibTorch (PyTorch C++ API)  |  '
        'All parameters: double precision (float64)  |  '
        'Total attention params: 16,641',
        fontsize=8.5, ha='center', va='center', fontfamily='serif',
        color='#888888',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9F9',
                  edgecolor='#D5D8DC', linewidth=0.8))

plt.tight_layout()
plt.savefig('RNN/lstm_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('RNN/lstm_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: RNN/lstm_architecture.png and RNN/lstm_architecture.pdf")
plt.show()
