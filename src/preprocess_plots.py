import numpy as np
import pandas as pd

def read_touchstone_reflection_db(filepath):
    """Reads a Touchstone file and extracts frequency (to GHz) and the reflection column (S11) in dB."""
    frequencies = []
    s_db = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('!', '#')):
                continue
            # Handle European comma decimal separators if present
            parts = line.replace(',', '.').split()
            if len(parts) >= 3:
                freq = float(parts[0]) / 1e9  # Convert Hz to GHz
                
                # Extract the reflection coefficient (Columns 1 & 2)
                real, imag = float(parts[1]), float(parts[2])
                mag = np.sqrt(real**2 + imag**2)
                val_db = 20 * np.log10(max(mag, 1e-6))
                
                frequencies.append(freq)
                s_db.append(val_db)
                
    return pd.DataFrame({'freq_ghz': frequencies, 'db': s_db})

def read_cst_txt(filepath):
    """Reads CST XY data exchange format file and converts complex none impedance to dB."""
    freqs, r, i = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                parts = line.replace(',', '.').split()
                freqs.append(float(parts[0]))
                r.append(float(parts[1]))
                i.append(float(parts[2]))
                
    mag = np.sqrt(np.array(r)**2 + np.array(i)**2)
    db = 20 * np.log10(np.clip(mag, 1e-6, None))
    return freqs, db

# 1. Process Measurements (Extracting reflection data from both files)
df_s11_meas = read_touchstone_reflection_db('/home/nedal/Desktop/RFID/WIN-VNA-Toolkit/data/2026-05-14/measurements/s11-s21.s2p')
df_s22_meas = read_touchstone_reflection_db('/home/nedal/Desktop/RFID/WIN-VNA-Toolkit/data/2026-05-14/measurements/s12-s22.s2p')

# 2. Process Simulations (Matching internal headers)
freq_s11_sim, s11_sim_db = read_cst_txt('/home/nedal/Desktop/RFID/WIN-VNA-Toolkit/data/2026-05-14/simulations/CST/S11.txt')  # Curvelabel = S1,1
freq_s22_sim, s22_sim_db = read_cst_txt('/home/nedal/Desktop/RFID/WIN-VNA-Toolkit/data/2026-05-14/simulations/CST/S22.txt')  # Curvelabel = S2,2

df_sim_s11 = pd.DataFrame({'freq_ghz': freq_s11_sim, 'S11_sim': s11_sim_db})
df_sim_s22 = pd.DataFrame({'freq_ghz': freq_s22_sim, 'S22_sim': s22_sim_db})

# 3. Create a clean grid of 1000 points across your sweep (0.75 GHz to 1.0 GHz)
target_freqs = np.linspace(0.75, 1.0, 1000)

s11_meas_interp = np.interp(target_freqs, df_s11_meas['freq_ghz'], df_s11_meas['db'])
s22_meas_interp = np.interp(target_freqs, df_s22_meas['freq_ghz'], df_s22_meas['db'])
s11_sim_interp = np.interp(target_freqs, df_sim_s11['freq_ghz'], df_sim_s11['S11_sim'])
s22_sim_interp = np.interp(target_freqs, df_sim_s22['freq_ghz'], df_sim_s22['S22_sim'])

# 4. Save to a standardized CSV
final_df = pd.DataFrame({
    'freq_ghz': target_freqs,
    'S11_sim': s11_sim_interp,
    'S11_meas': s11_meas_interp,
    'S22_sim': s22_sim_interp,
    'S22_meas': s22_meas_interp
})

final_df.to_csv('antenna_mapping_results.csv', index=False)
print("Corrected CSV 'antenna_mapping_results.csv' generated!")