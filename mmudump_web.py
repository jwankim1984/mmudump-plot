"""
MMU Dump Plot - Streamlit Web Application
DMRS SNR Calculator for 5G NR and LTE
"""

import os
import io
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ==================== 3GPP TS 38.211 DMRS Functions (NR) ====================

def gold_sequence(c_init, length):
    """Generate Gold sequence (3GPP TS 38.211 Section 5.2.1)"""
    N = length + 1600
    x1 = np.zeros(N + 31, dtype=np.uint32)
    x2 = np.zeros(N + 31, dtype=np.uint32)
    
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    
    for n in range(31, N + 31):
        x1[n] = (x1[n - 28] + x1[n - 31]) % 2
        x2[n] = (x2[n - 28] + x2[n - 29] + x2[n - 30] + x2[n - 31]) % 2
    
    c = (x1[1600:1600+length] + x2[1600:1600+length]) % 2
    return c

def generate_nr_dmrs_type1_3gpp(n_rb, start_rb, slot, symbol_idx, n_id, n_scid=0, cdm_group=0):
    """Generate NR DMRS sequence for Type 1 (3GPP TS 38.211 Section 7.4.1.1.1)"""
    m = n_rb * 6
    l = symbol_idx
    c_init = (2**17 * (14 * slot + l + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (2**31)
    
    c = gold_sequence(c_init, 2 * m).astype(np.int32)
    
    dmrs_seq = np.zeros(m, dtype=complex)
    for i in range(m):
        i_val = (1 - 2 * c[2*i]) / np.sqrt(2)
        q_val = (1 - 2 * c[2*i + 1]) / np.sqrt(2)
        dmrs_seq[i] = i_val + 1j * q_val
    
    k_prime = cdm_group
    dmrs_indices = []
    for rb in range(n_rb):
        rb_start = (start_rb + rb) * 12
        for m_val in range(6):
            k = rb_start + k_prime + 2 * m_val
            dmrs_indices.append(k)
    
    return dmrs_seq, np.array(dmrs_indices)

def generate_nr_dmrs_type2_3gpp(n_rb, start_rb, slot, symbol_idx, n_id, n_scid=0, cdm_group=0):
    """Generate NR DMRS sequence for Type 2 (3GPP TS 38.211 Section 7.4.1.1.1)"""
    m = n_rb * 4
    l = symbol_idx
    c_init = (2**17 * (14 * slot + l + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (2**31)
    
    c = gold_sequence(c_init, 2 * m).astype(np.int32)
    
    dmrs_seq = np.zeros(m, dtype=complex)
    for i in range(m):
        i_val = (1 - 2 * c[2*i]) / np.sqrt(2)
        q_val = (1 - 2 * c[2*i + 1]) / np.sqrt(2)
        dmrs_seq[i] = i_val + 1j * q_val
    
    dmrs_indices = []
    for rb in range(n_rb):
        rb_start = (start_rb + rb) * 12
        if cdm_group == 0:
            dmrs_indices.extend([rb_start + 0, rb_start + 1])
        elif cdm_group == 1:
            dmrs_indices.extend([rb_start + 6, rb_start + 7])
        elif cdm_group == 2:
            dmrs_indices.extend([rb_start + 2, rb_start + 3, rb_start + 8, rb_start + 9])
    
    return dmrs_seq, np.array(dmrs_indices)

def calculate_dmrs_snr_3gpp(rx_dmrs_symbol, ideal_dmrs, dmrs_indices):
    """Calculate SNR using ideal DMRS sequence (3GPP standard)"""
    rx_dmrs = rx_dmrs_symbol[dmrs_indices]
    H = rx_dmrs / ideal_dmrs
    H_smooth = np.convolve(H, np.ones(5)/5, mode='same')
    noise_est = H - H_smooth
    noise_power = np.mean(np.abs(noise_est) ** 2)
    signal_power = np.mean(np.abs(H_smooth) ** 2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db, signal_power, noise_power

# ==================== LTE DMRS Functions (3GPP TS 36.211) ====================

def generate_lte_dmrs(n_rb, start_rb, slot, symbol_idx, n_id, v_shift=0):
    """Generate LTE PUSCH DMRS sequence (3GPP TS 36.211 Section 5.5.1, 5.5.2)
    
    For PUSCH DMRS:
    - Base sequence r_u,v(n) is generated based on group number u and sequence number v
    - Cyclic shift alpha is applied
    - For M_sc >= 36 (3 RBs), Zadoff-Chu based sequence
    - For M_sc < 36, special sequence
    
    LTE slot structure:
    - Symbol 0-6: first slot (ns = slot * 2)
    - Symbol 7-13: second slot (ns = slot * 2 + 1)
    """
    M_sc = n_rb * 12  # Number of subcarriers
    
    # Calculate ns (slot number within radio frame)
    # LTE: Each subframe has 2 slots (0.5ms each)
    # Symbol 0-6 -> ns = slot * 2
    # Symbol 7-13 -> ns = slot * 2 + 1
    if symbol_idx <= 6:
        ns = slot * 2
    else:
        ns = slot * 2 + 1
    
    # Group hopping disabled by default (simplified)
    # When group hopping is disabled: u = f_ss
    # f_ss = n_ID mod 30
    u = n_id % 30
    
    # Sequence number v (3GPP TS 36.211 Section 5.5.1.3)
    # v = 0 for base sequence
    v = 0
    
    # Generate base sequence r_u,v(n) (3GPP TS 36.211 Section 5.5.1.2)
    if M_sc >= 36:
        # For M_sc >= 3*N_sc^RB = 36: Use Zadoff-Chu based sequence
        # Find largest prime less than M_sc
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
        N_zc = M_sc
        for p in reversed(primes):
            if p < M_sc:
                N_zc = p
                break
        
        # Zadoff-Chu root index q (3GPP TS 36.211 Section 5.5.1.3)
        # q = floor(N_zc * (u + 1) / 31)
        q = (N_zc * (u + 1)) // 31
        
        # Generate Zadoff-Chu root sequence
        n = np.arange(N_zc)
        zc_seq = np.exp(-1j * np.pi * q * n * (n + 1) / N_zc)
        
        # Cyclic extension to M_sc
        r_uv = np.zeros(M_sc, dtype=complex)
        r_uv[:N_zc] = zc_seq
        r_uv[N_zc:] = zc_seq[:M_sc - N_zc]  # Cyclic extension
    else:
        # For M_sc < 36: Use sequence from table (5.5.1.2-2)
        # Simplified: use QPSK-like sequence
        c_init = (n_id + 1) * (2 * ns + 1) * 2**16 + n_id
        c_init = c_init % (2**31)
        c = gold_sequence(c_init, 2 * M_sc).astype(np.int32)
        
        r_uv = np.zeros(M_sc, dtype=complex)
        for i in range(M_sc):
            i_val = (1 - 2 * c[2*i]) / np.sqrt(2)
            q_val = (1 - 2 * c[2*i + 1]) / np.sqrt(2)
            r_uv[i] = i_val + 1j * q_val
    
    # Apply cyclic shift alpha (3GPP TS 36.211 Section 5.5.2.1.1)
    # For PUSCH DMRS: n_cs from higher layer, default 0
    n_cs = v_shift
    alpha = 2 * np.pi * n_cs / 12
    
    # DMRS sequence: r(n) = r_uv(n) * exp(j * alpha * n)
    dmrs_seq = r_uv * np.exp(1j * alpha * np.arange(M_sc))
    
    dmrs_indices = np.arange(start_rb * 12, (start_rb + n_rb) * 12)
    
    return dmrs_seq, dmrs_indices

def calculate_lte_dmrs_snr(rx_dmrs_symbol, ideal_dmrs, dmrs_indices):
    """Calculate SNR using LTE DMRS sequence"""
    rx_dmrs = rx_dmrs_symbol[dmrs_indices]
    H = rx_dmrs / ideal_dmrs
    H_smooth = np.convolve(H, np.ones(5)/5, mode='same')
    noise_est = H - H_smooth
    noise_power = np.mean(np.abs(noise_est) ** 2)
    signal_power = np.mean(np.abs(H_smooth) ** 2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db, signal_power, noise_power

# ==================== Data Processing Functions ====================

def swap_64_by_8(data):
    """Swap bytes within each 8-byte (64-bit) group"""
    arr = np.frombuffer(data, dtype=np.uint64)
    return arr.byteswap().tobytes()

def convert_real_imag(A):
    """Convert binary data to complex numbers (I + jQ)"""
    A = np.asarray(A, dtype=np.uint32)
    i_vals = (A & 0xffff).astype(np.int16)
    q_vals = ((A >> 16) & 0xffff).astype(np.int16)
    return i_vals.astype(np.float64) + 1j * q_vals.astype(np.float64)

def load_mbin_file(filepath):
    """Load and parse .mbin file"""
    with open(filepath, 'rb') as FP:
        raw = FP.read()
    swapped = swap_64_by_8(raw)
    A = np.frombuffer(swapped, dtype=np.uint32)
    B = convert_real_imag(A)
    return B

def load_txt_file(filepath):
    """Load IQ data from text file"""
    data = np.loadtxt(filepath, dtype=np.float64)
    i_vals = data[:, 0]
    q_vals = data[:, 1]
    iq_data = i_vals + 1j * q_vals
    
    NumSym = 14
    TonePerRB = 12
    total_samples = len(iq_data)
    samples_per_slot = total_samples // NumSym
    calculated_num_rb = samples_per_slot // TonePerRB
    
    return iq_data, calculated_num_rb

def get_slot_for_dmrs(slot, scs):
    """Get slot number for DMRS sequence generation based on SCS"""
    if scs == "15kHz":
        slots_per_frame = 10
    elif scs == "30kHz":
        slots_per_frame = 20
    elif scs == "60kHz":
        slots_per_frame = 40
    elif scs == "120kHz":
        slots_per_frame = 80
    else:
        slots_per_frame = 20
    
    return slot % slots_per_frame

def perform_dft(data, n_fft=None):
    """Perform DFT on the allocated RB data"""
    if n_fft is None:
        n_fft = len(data)
    return np.fft.fftshift(np.fft.fft(data, n_fft))

# ==================== Streamlit App ====================

st.set_page_config(page_title="MMU Dump Plot", page_icon="📊", layout="wide")

st.title("📊 MMU Dump Plot - DMRS SNR Calculator")
st.markdown("### 5G NR / LTE DMRS Signal Analysis Tool")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    st.subheader("📁 File Selection")
    uploaded_file = st.file_uploader("Upload IQ Data File", type=['mbin', 'txt'])
    
    # System Mode
    st.subheader("📶 System Mode")
    mode = st.radio("Select Mode", ["NR (5G)", "LTE (4G)"])
    mode = "NR" if "NR" in mode else "LTE"
    
    # SCS
    st.subheader("📡 SCS")
    scs = st.selectbox("Subcarrier Spacing", ["15kHz", "30kHz", "60kHz", "120kHz"])
    
    # DMRS Settings
    st.subheader("🔧 DMRS Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        pci = st.number_input("PCI", min_value=0, max_value=1007, value=1)
        cdm_group = st.selectbox("CDM Group", ["0", "1", "2"])
        slot = st.number_input("Slot", min_value=0, max_value=159, value=4)
    
    with col2:
        dmrs_type = st.selectbox("DMRS Type", ["Type 1", "Type 2"])
        start_rb = st.number_input("Start RB", min_value=0, max_value=272, value=0)
        rb_size = st.number_input("RB Size", min_value=1, max_value=273, value=51)
    
    num_rb = st.number_input("NumRB (Total)", min_value=1, max_value=273, value=273)
    
    dmrs_syms_str = st.text_input("DMRS Symbols", value="2, 11")
    dmrs_syms = [int(x.strip()) for x in dmrs_syms_str.split(',')]

# Main content
if uploaded_file is not None:
    # Load data directly from uploaded file (no temp file needed for txt)
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext == '.txt':
            # For txt files, read directly from BytesIO
            content = uploaded_file.read().decode('utf-8')
            data_lines = np.array([list(map(float, line.strip().split())) for line in content.strip().split('\n') if line.strip()])
            i_vals = data_lines[:, 0]
            q_vals = data_lines[:, 1]
            raw_data = i_vals + 1j * q_vals
            
            NumSym = 14
            TonePerRB = 12
            total_samples = len(raw_data)
            samples_per_slot = total_samples // NumSym
            calculated_num_rb = samples_per_slot // TonePerRB
            num_rb = calculated_num_rb
            
            st.info(f"Detected TXT format. Auto-detected NumRB: {calculated_num_rb}")
        else:
            # For mbin files, use tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mbin') as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            raw_data = load_mbin_file(temp_path)
            os.unlink(temp_path)
            st.info(f"Detected MBIN format.")
        
        NumSym = 14
        TonePerRB = 12
        NumTone = num_rb * TonePerRB
        
        SlotCnt = len(raw_data) // (NumTone * NumSym)
        
        if SlotCnt == 0:
            C = raw_data[:NumTone * NumSym].reshape(1, NumTone * NumSym)
            SlotCnt = 1
        else:
            C = raw_data[:SlotCnt * NumTone * NumSym].reshape(SlotCnt, NumTone * NumSym)
        
        st.success(f"File loaded: {SlotCnt} slots, {NumTone} tones ({num_rb} RBs), {NumSym} symbols")
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        run_calc = st.button("🚀 Run SNR Calculation", type="primary")
    with col2:
        plot_iq = st.button("📈 IQ Constellation")
    with col3:
        plot_mag = st.button("📊 Magnitude")
    with col4:
        plot_3d = st.button("🌊 3D Waveform")
    
    # Export button
    export_iq = st.button("💾 Export IQ to TXT")
    
    # Run calculation
    if run_calc:
        with st.spinner("Calculating..."):
            # Get modular slot
            dmrs_slot = get_slot_for_dmrs(slot, scs)
            
            # Get data
            data_slot_idx = 0 if C.shape[0] == 1 else slot
            data = C[data_slot_idx, :]
            dat_sym = data.reshape(NumSym, NumTone).T
            
            n_id = pci
            cdm_group_int = int(cdm_group)
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                st.markdown(f"**PCI:** {pci}")
                st.markdown(f"**Slot:** {slot}")
                st.markdown(f"**SCS:** {scs}")
                st.markdown(f"**DMRS Slot (modular):** {dmrs_slot}")
                st.markdown(f"**CDM Group:** {cdm_group_int}")
                st.markdown(f"**Start RB:** {start_rb}")
                st.markdown(f"**RB Size:** {rb_size}")
                st.markdown(f"**DMRS Symbols:** {dmrs_syms}")
                
                if mode == "NR":
                    st.markdown(f"**Mode:** NR, DMRS Type: {dmrs_type}")
                else:
                    st.markdown(f"**Mode:** LTE")
            
            # Generate DMRS and calculate SNR
            start_idx = start_rb * 12
            end_idx = (start_rb + rb_size) * 12
            
            if mode == "NR":
                if dmrs_type == "Type 1":
                    generate_dmrs = generate_nr_dmrs_type1_3gpp
                else:
                    generate_dmrs = generate_nr_dmrs_type2_3gpp
                
                H_dmrs_dict = {}
                snr_results = []
                for dmrs_sym in dmrs_syms:
                    rx_dmrs = dat_sym[:, dmrs_sym]
                    ideal_dmrs, dmrs_indices = generate_dmrs(
                        n_rb=rb_size, start_rb=start_rb, slot=dmrs_slot,
                        symbol_idx=dmrs_sym, n_id=n_id, cdm_group=cdm_group_int
                    )
                    snr_3gpp, _, _ = calculate_dmrs_snr_3gpp(rx_dmrs, ideal_dmrs, dmrs_indices)
                    snr_results.append((dmrs_sym, snr_3gpp))
                    H_dmrs_dict[dmrs_sym] = rx_dmrs[dmrs_indices] / ideal_dmrs
                
                ideal_dmrs, dmrs_indices = generate_dmrs(
                    n_rb=rb_size, start_rb=start_rb, slot=dmrs_slot,
                    symbol_idx=dmrs_syms[0], n_id=n_id, cdm_group=cdm_group_int
                )
            else:
                H_dmrs_dict = {}
                snr_results = []
                for dmrs_sym in dmrs_syms:
                    # LTE PUSCH: DMRS is already in frequency domain (no DFT was applied at TX)
                    rx_dmrs = dat_sym[start_idx:end_idx, dmrs_sym]  # Already in frequency domain
                    
                    ideal_dmrs, dmrs_indices = generate_lte_dmrs(
                        n_rb=rb_size, start_rb=start_rb, slot=dmrs_slot,
                        symbol_idx=dmrs_sym, n_id=n_id
                    )
                    snr_lte, _, _ = calculate_lte_dmrs_snr(rx_dmrs, ideal_dmrs, dmrs_indices)
                    snr_results.append((dmrs_sym, snr_lte))
                    H_dmrs_dict[dmrs_sym] = rx_dmrs / ideal_dmrs
                
                ideal_dmrs, dmrs_indices = generate_lte_dmrs(
                    n_rb=rb_size, start_rb=start_rb, slot=dmrs_slot,
                    symbol_idx=dmrs_syms[0], n_id=n_id
                )
            
            # Display SNR results
            with result_col1:
                st.markdown("### 📡 SNR Results")
                for sym, snr in snr_results:
                    st.metric(f"Symbol {sym} SNR", f"{snr:.2f} dB")
            
            # EQ Constellation plot
            start_idx = start_rb * 12
            end_idx = (start_rb + rb_size) * 12
            H_dmrs = H_dmrs_dict[dmrs_syms[0]]
            
            H_full = np.ones(end_idx - start_idx, dtype=complex)
            for i, idx in enumerate(dmrs_indices):
                local_idx = idx - start_idx
                if 0 <= local_idx < len(H_full):
                    H_full[local_idx] = H_dmrs[i]
            
            for k in range(1, len(H_full), 2):
                H_full[k] = H_full[k - 1]
            
            # Create plot
            with result_col2:
                fig, axes = plt.subplots(2, 7, figsize=(14, 8))
                fig.suptitle(f'EQ Constellation per Symbol (Slot {slot}, RB {start_rb}~{start_rb + rb_size - 1})')
                
                eq_all = []
                for sym in range(NumSym):
                    ax = axes[sym // 7, sym % 7]
                    rx_data = dat_sym[start_idx:end_idx, sym]
                    
                    if sym in dmrs_syms:
                        ax.scatter(np.real(rx_data[::2]), np.imag(rx_data[::2]), marker='.', s=1)
                        ax.set_title(f'DMRS Sym {sym}', fontsize=8, color='red')
                    else:
                        eq_data = rx_data / H_full
                        eq_all.append(eq_data)
                        ax.scatter(np.real(eq_data), np.imag(eq_data), marker='.', s=1)
                        ax.set_title(f'Data Sym {sym}', fontsize=8)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate EVM
                if eq_all:
                    eq_all = np.concatenate(eq_all)
                    eq_normalized = eq_all / np.abs(eq_all)
                    qpsk_points = (1 + 1j) / np.sqrt(2) * np.array([1+0j, 1j, -1+0j, -1j])
                    evm = [np.min(np.abs(pt - qpsk_points)) ** 2 for pt in eq_normalized]
                    evm_rms = np.sqrt(np.mean(evm)) * 100
                    evm_db = 20 * np.log10(evm_rms / 100)
                    
                    st.metric("EVM (QPSK)", f"{evm_rms:.2f}% ({evm_db:.2f} dB)")
    
    # IQ Constellation plot
    if plot_iq:
        st.subheader("📈 IQ Constellation")
        
        data_slot_idx = 0 if C.shape[0] == 1 else slot
        dat_sym = C[data_slot_idx, :].reshape(NumSym, NumTone).T
        
        fig, axes = plt.subplots(2, 7, figsize=(14, 8))
        
        if mode == "NR":
            plot_data = dat_sym
            title_suffix = 'Raw (Full BW)'
        else:
            start_idx = start_rb * 12
            end_idx = (start_rb + rb_size) * 12
            plot_data = dat_sym[start_idx:end_idx, :]
            title_suffix = f'DFT (RB {start_rb}~{start_rb + rb_size - 1})'
        
        fig.suptitle(f'Slot {slot} - IQ Constellation ({title_suffix})')
        
        for sym in range(NumSym):
            ax = axes[sym // 7, sym % 7]
            
            if mode == "LTE":
                if sym in dmrs_syms:
                    sym_data = plot_data[:, sym]
                else:
                    sym_data = perform_dft(plot_data[:, sym])
            else:
                sym_data = plot_data[:, sym]
            
            if sym in dmrs_syms:
                ax.scatter(np.real(sym_data), np.imag(sym_data), marker='.', s=1)
                ax.set_title(f'DMRS Sym {sym}', fontsize=8, color='red')
            else:
                ax.scatter(np.real(sym_data), np.imag(sym_data), marker='.', s=1)
                ax.set_title(f'Data Sym {sym}', fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Magnitude plot
    if plot_mag:
        st.subheader("📊 Magnitude")
        
        data_slot_idx = 0 if C.shape[0] == 1 else slot
        dat_sym = C[data_slot_idx, :].reshape(NumSym, NumTone).T
        
        fig, axes = plt.subplots(2, 7, figsize=(14, 8))
        fig.suptitle(f'Slot {slot} - Magnitude (Full BW)')
        
        for sym in range(NumSym):
            ax = axes[sym // 7, sym % 7]
            mag_sym = np.abs(dat_sym[:, sym])
            
            ax.plot(np.arange(len(mag_sym)), mag_sym)
            if sym in dmrs_syms:
                ax.set_title(f'DMRS Sym {sym}', fontsize=8, color='red')
            else:
                ax.set_title(f'Data Sym {sym}', fontsize=8)
            ax.set_xlabel('Subcarrier', fontsize=6)
            ax.set_ylabel('Mag', fontsize=6)
            ax.tick_params(axis='both', labelsize=6)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 3D Waveform plot
    if plot_3d:
        st.subheader("🌊 3D Waveform")
        
        data_slot_idx = 0 if C.shape[0] == 1 else slot
        dat_sym = C[data_slot_idx, :].reshape(NumSym, NumTone).T
        
        mag = np.abs(dat_sym)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(np.arange(NumSym), np.arange(NumTone))
        ax.plot_surface(X, Y, mag, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Time (Symbol)')
        ax.set_ylabel('Subcarrier')
        ax.set_zlabel('Magnitude')
        ax.set_title(f'Slot {slot} - 3D Waveform (Full Bandwidth)')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Export IQ to TXT
    if export_iq:
        st.subheader("💾 Export IQ to TXT")
        
        data_slot_idx = 0 if C.shape[0] == 1 else slot
        dat_sym = C[data_slot_idx, :].reshape(NumSym, NumTone).T
        
        # Create TXT content in memory
        output = io.StringIO()
        for sym in range(NumSym):
            for tone in range(NumTone):
                i_val = int(np.round(np.real(dat_sym[tone, sym])))
                q_val = int(np.round(np.imag(dat_sym[tone, sym])))
                output.write(f"{i_val} {q_val}\n")
        
        output.seek(0)
        
        st.success(f"IQ data ready for export: {NumSym} symbols × {NumTone} tones = {NumSym * NumTone} lines")
        
        # Download button
        st.download_button(
            label="📥 Download IQ Data (TXT)",
            data=output.getvalue(),
            file_name=f"iq_export_slot{slot}.txt",
            mime="text/plain"
        )
    
    # No cleanup needed for txt files (read directly)
    # mbin temp file already cleaned up after reading

else:
    st.info("👆 Please upload an IQ data file (.mbin or .txt) to begin analysis.")
    
    st.markdown("""
    ### 📋 File Format
    
    **TXT Format:**
    - Each line contains: `I Q` (decimal values)
    - Symbol-major order: sym0_tone0, sym0_tone1, ..., sym13_toneN
    - Example: `-184 1720`
    
    **MBIN Format:**
    - Binary format with 64-bit word swapping
    - Each 32-bit word: Q (upper 16 bits) + I (lower 16 bits)
    
    ### 🚀 Features
    - NR (5G) and LTE (4G) support
    - DMRS Type 1 / Type 2 (NR)
    - SCS: 15kHz, 30kHz, 60kHz, 120kHz
    - SNR calculation per DMRS symbol
    - EQ Constellation visualization
    - EVM calculation
    """)

# Footer
st.markdown("---")
st.markdown("*MMU Dump Plot - DMRS SNR Calculator | 3GPP TS 38.211 / TS 36.211 Compliant*")
st.markdown("📚 [MMU Dump Plot Guidance](https://confluence-nw.sec.samsung.net/spaces/MODEMSOC/pages/894937373/MMU+Dump+Plot+Guidance)")
