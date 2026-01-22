import matplotlib.pyplot as plt
import os

# Data matching your recent test results
metrics = {
    'Compression': 97.9,
    'Discrepancies': 7,
    'Healed': 2
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Aether System Performance Snapshot', fontsize=14)

# Plot 1: Compression
axes[0].bar(['Compressed'], [metrics['Compression']], color='green')
axes[0].set_ylim(0, 100)
axes[0].set_title(f"Storage Efficiency: {metrics['Compression']}%")
axes[0].set_ylabel('Percentage Saved')

# Plot 2: Healing
axes[1].bar(['Found', 'Healed'], [metrics['Discrepancies'], metrics['Healed']], color=['orange', 'blue'])
axes[1].set_title('Auto-Healing Operations')

plt.tight_layout()

# CORRECTED PATH: Saves to the folder you just created in the project root
plt.savefig('docs/screenshots/dashboard_snapshot.png')
print("Graph generated at docs/screenshots/dashboard_snapshot.png")