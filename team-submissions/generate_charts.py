"""
Generate Visualization Charts for NVIDIA iQuHACK 2026 Presentation

Creates charts showing:
1. Quantum Circuit Time vs N
2. MTS Time vs N
3. CPU vs GPU comparison
4. Memory scaling visualization
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Benchmark data
quantum_data = {
    'N': [20, 25, 30],
    'time': [7.77, 13.50, 61.68],
    'platform': ['GPU (Brev L4)', 'GPU (Brev L4)', 'CPU (qBraid)']
}

mts_data = {
    'N': [20, 25, 30, 35, 40],
    'time': [1.08, 2.40, 6.39, 10.56, 16.52],
    'energy': [26, 48, 83, 97, 128]
}

# Memory requirements (theoretical)
memory_data = {
    'N': [20, 25, 30, 35, 40],
    'memory_gb': [0.016, 0.5, 16, 550, 16000]
}

def create_quantum_time_chart():
    """Create Quantum Circuit Time vs N chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#76B900', '#76B900', '#0077B6']  # NVIDIA green for GPU, blue for CPU
    
    bars = ax.bar(quantum_data['N'], quantum_data['time'], color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Problem Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Quantum Circuit Execution Time vs Problem Size', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for bar, time, platform in zip(bars, quantum_data['time'], quantum_data['platform']):
        height = bar.get_height()
        ax.annotate(f'{time:.2f}s\n({platform})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(quantum_data['N'])
    ax.set_ylim(0, max(quantum_data['time']) * 1.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#76B900', label='GPU (Brev L4)'),
                      Patch(facecolor='#0077B6', label='CPU (qBraid)')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('chart_quantum_time.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_quantum_time.png")
    plt.close()

def create_mts_time_chart():
    """Create MTS Time vs N chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(mts_data['N'], mts_data['time'], 'o-', color='#76B900', linewidth=2, markersize=10, label='MTS Time')
    
    ax.set_xlabel('Problem Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Classical MTS Algorithm Time vs Problem Size', fontsize=16, fontweight='bold')
    
    # Add value labels
    for n, t in zip(mts_data['N'], mts_data['time']):
        ax.annotate(f'{t:.2f}s', xy=(n, t), xytext=(5, 5),
                   textcoords="offset points", fontsize=10)
    
    ax.set_xticks(mts_data['N'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('chart_mts_time.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_mts_time.png")
    plt.close()

def create_energy_chart():
    """Create Best Energy vs N chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(mts_data['N'], mts_data['energy'], color='#E63946', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Problem Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Energy Found', fontsize=14, fontweight='bold')
    ax.set_title('Best LABS Energy Found by MTS Algorithm', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, (n, e) in enumerate(zip(mts_data['N'], mts_data['energy'])):
        ax.annotate(f'{e}', xy=(n, e), xytext=(0, 3),
                   textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks(mts_data['N'])
    
    plt.tight_layout()
    plt.savefig('chart_energy.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_energy.png")
    plt.close()

def create_memory_chart():
    """Create Memory Requirements chart with log scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#76B900', '#76B900', '#FFA500', '#E63946', '#E63946']
    
    bars = ax.bar(memory_data['N'], memory_data['memory_gb'], color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Problem Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Memory Required (GB) - Log Scale', fontsize=14, fontweight='bold')
    ax.set_title('Quantum Simulation Memory Requirements', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    
    # Add horizontal lines for GPU memory limits
    ax.axhline(y=24, color='blue', linestyle='--', linewidth=2, label='L4 GPU (24GB)')
    ax.axhline(y=80, color='purple', linestyle='--', linewidth=2, label='A100 GPU (80GB)')
    
    # Add labels
    labels = ['✅ 16MB', '✅ 0.5GB', '⚠️ 16GB', '❌ 550GB', '❌ 16TB']
    for bar, lbl in zip(bars, labels):
        height = bar.get_height()
        ax.annotate(lbl, xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    
    ax.set_xticks(memory_data['N'])
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('chart_memory.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_memory.png")
    plt.close()

def create_combined_comparison():
    """Create combined CPU vs GPU comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Time comparison
    n_vals = [20, 25, 30]
    gpu_times = [7.77, 13.50, None]  # N=30 not done on GPU
    cpu_times = [None, None, 61.68]  # Only N=30 on CPU
    
    x = np.arange(len(n_vals))
    width = 0.35
    
    ax1.bar(x - width/2, [7.77, 13.50, 0], width, label='GPU (Brev L4)', color='#76B900')
    ax1.bar(x + width/2, [0, 0, 61.68], width, label='CPU (qBraid)', color='#0077B6')
    
    ax1.set_xlabel('Problem Size (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Quantum Circuit: GPU vs CPU', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_vals)
    ax1.legend()
    
    # Chart 2: MTS scaling
    ax2.plot(mts_data['N'], mts_data['time'], 'o-', color='#76B900', linewidth=2, markersize=10)
    ax2.set_xlabel('Problem Size (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Classical MTS: Linear Scaling', fontsize=14, fontweight='bold')
    ax2.set_xticks(mts_data['N'])
    
    plt.tight_layout()
    plt.savefig('chart_combined.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_combined.png")
    plt.close()

if __name__ == "__main__":
    print("Generating visualization charts...")
    create_quantum_time_chart()
    create_mts_time_chart()
    create_energy_chart()
    create_memory_chart()
    create_combined_comparison()
    print("\nAll charts generated! Use these in your presentation.")
