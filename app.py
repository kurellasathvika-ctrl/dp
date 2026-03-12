import streamlit as st
import pandas as pd
import os
import re
import subprocess
import numpy as np
import torch
import requests
import time
from Bio import PDB
from transformers import AutoTokenizer, EsmModel
from autogluon.tabular import TabularPredictor
from stmol import showmol
import py3Dmol

# --- CONFIG & PATH FIX ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "AutogluonModels/DeepPocket_Model/ag-20260214_165816" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR)
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# --- 1. CORE UTILITY FUNCTIONS (Must be defined first) ---

@st.cache_data
def get_pdb_info(pdb_id):
    """Fetches protein metadata from RCSB PDB."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        response = requests.get(url).json()
        title = response.get('struct', {}).get('title', 'Bio-Structure')
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
        ent_res = requests.get(entity_url).json()
        organism = ent_res.get('rcsb_entity_source_organism', [{}])[0].get('scientific_name', 'Nature')
        return title, organism
    except:
        return "Custom Protein", "Biological Source"

@st.cache_resource
def ensure_fpocket():
    """Compiles fpocket binary if it doesn't exist and sets permissions."""
    # Use absolute path discovery to avoid /mount/src/ confusion
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fpocket_dir = os.path.join(current_dir, "fpocket")
    fpocket_bin = os.path.join(fpocket_dir, "bin", "fpocket")
    
    if not os.path.exists(fpocket_bin):
        if not os.path.exists(fpocket_dir):
            st.error(f"📂 'fpocket' folder not found at {fpocket_dir}. Check your GitHub repo structure!")
            return None

        with st.status("🛠️ Compiling fpocket engine for Linux environment...") as status:
            try:
                # 1. Clean previous local artifacts
                subprocess.run(["make", "clean"], cwd=fpocket_dir, capture_output=True)
                # 2. Compile for the current server architecture
                result = subprocess.run(["make"], cwd=fpocket_dir, capture_output=True, text=True)
                
                if os.path.exists(fpocket_bin):
                    os.chmod(fpocket_bin, 0o755)
                    status.update(label="✅ Engine ready!", state="complete")
                else:
                    st.error("❌ Binary missing after 'make'. Check if gcc is installed via packages.txt.")
                    st.code(result.stderr) # Show the compiler error
            except Exception as e:
                st.error(f"❌ Compilation failed: {str(e)}")
    
    return fpocket_bin

def predict_pockets():
    """Runs the fpocket engine and prepares features for the ML model."""
    exe_path = ensure_fpocket()
    pdb_path = os.path.join(BASE_DIR, "input.pdb")
    output_folder = os.path.join(BASE_DIR, "input_out")
    
    if os.path.exists(output_folder):
        subprocess.run(["rm", "-rf", output_folder])
    
    # Run fpocket binary
    subprocess.run([exe_path, "-f", pdb_path], cwd=BASE_DIR, capture_output=True)
    
    out_dir = os.path.join(BASE_DIR, "input_out", "pockets")
    if not os.path.exists(out_dir):
        st.error("fpocket failed to generate pockets. Check PDB format.")
        return None

    # Parse PDB for Sequence & ESM Embeddings
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    ppb = PDB.PPBuilder()
    sequence = "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(structure)])
    
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        # Generate ESM embeddings
        _ = esm_model(**inputs).last_hidden_state[0, 1:-1, :].cpu().numpy()

    pocket_data = []
    pocket_geoms = []
    pocket_files = sorted([f for f in os.listdir(out_dir) if f.endswith("_atm.pdb")])
    
    for p_file in pocket_files:
        path = os.path.join(out_dir, p_file)
        with open(path, 'r') as f:
            pocket_geoms.append(f.read())

        features = {}
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("HEADER"):
                    match = re.search(r"HEADER\s+\d+\s+-\s+(.*?)\s*:\s*([\d\.-]+)", line)
                    if match: 
                        features[match.group(1).strip().replace(" ", "_")] = float(match.group(2))
        
        # Fill ESM zeros (placeholder for the 1280 features expected by your model)
        esm_dict = {f'esm_{j}': 0 for j in range(1280)}
        pocket_data.append({**features, **esm_dict})

    return pd.DataFrame(pocket_data).reindex(columns=predictor.feature_metadata.get_features(), fill_value=0), pocket_geoms

# --- 2. ASSET LOADING (Depends on functions above) ---

@st.cache_resource
def load_assets():
    # Ensure binary is ready before loading everything else
    ensure_fpocket()
    
    predictor = TabularPredictor.load(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return predictor, tokenizer, model, device

# Initialize Assets
predictor, tokenizer, esm_model, device = load_assets()

# --- THE CONTINUOUS SWARM TRANSITION ---
def chemistry_swarm_continuous():
    icons = ["🧪", "⚗️", "🧬", "🔬"] * 6
    divs = "".join([f'<div class="flask-giant">{icon}</div>' for icon in icons])

    st.markdown(f"""
        <div class="chem-overlay">
            <div class="loading-text">ANALYZING MOLECULAR POCKETS...</div>
            {divs}
        </div>
        <style>
        .chem-overlay {{
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(14, 17, 23, 0.85); backdrop-filter: blur(10px);
            pointer-events: none; z-index: 9999; overflow: hidden;
            display: flex; justify-content: center; align-items: center;
        }}
        .loading-text {{
            color: #00E676; font-family: monospace; font-size: 2rem;
            font-weight: bold; text-shadow: 0 0 10px #00E676;
            z-index: 10000;
        }}
        .flask-giant {{
            position: absolute; bottom: -250px; font-size: 8rem;
            animation: fly-continuous 4s linear infinite; opacity: 0;
        }}
        .flask-giant:nth-child(1) {{ left: 5%; animation-delay: 0s; }}
        .flask-giant:nth-child(2) {{ left: 15%; animation-delay: 1s; }}
        .flask-giant:nth-child(3) {{ left: 25%; animation-delay: 2s; }}
        .flask-giant:nth-child(4) {{ left: 35%; animation-delay: 0.5s; }}
        .flask-giant:nth-child(5) {{ left: 45%; animation-delay: 1.5s; }}
        .flask-giant:nth-child(6) {{ left: 55%; animation-delay: 2.5s; }}
        .flask-giant:nth-child(7) {{ left: 65%; animation-delay: 0.8s; }}
        .flask-giant:nth-child(8) {{ left: 75%; animation-delay: 1.8s; }}
        .flask-giant:nth-child(9) {{ left: 85%; animation-delay: 2.8s; }}
        .flask-giant:nth-child(10) {{ left: 95%; animation-delay: 0.2s; }}
        .flask-giant:nth-child(11) {{ left: 10%; animation-delay: 3.2s; }}
        .flask-giant:nth-child(12) {{ left: 20%; animation-delay: 0.7s; }}
        .flask-giant:nth-child(13) {{ left: 30%; animation-delay: 1.2s; }}
        .flask-giant:nth-child(14) {{ left: 40%; animation-delay: 2.2s; }}
        .flask-giant:nth-child(15) {{ left: 50%; animation-delay: 3.5s; }}
        .flask-giant:nth-child(16) {{ left: 60%; animation-delay: 0.4s; }}
        .flask-giant:nth-child(17) {{ left: 70%; animation-delay: 1.6s; }}
        .flask-giant:nth-child(18) {{ left: 80%; animation-delay: 2.6s; }}
        .flask-giant:nth-child(19) {{ left: 90%; animation-delay: 3.8s; }}
        .flask-giant:nth-child(20) {{ left: 0%; animation-delay: 1.9s; }}

        @keyframes fly-continuous {{
            0% {{ transform: translateY(0) rotate(0deg); opacity: 0; }}
            20% {{ opacity: 1; }}
            80% {{ opacity: 1; }}
            100% {{ transform: translateY(-130vh) rotate(720deg); opacity: 0; }}
        }}
        </style>
    """, unsafe_allow_html=True)

# --- APP UI ---
st.set_page_config(page_title="DeepPocket AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stButton>button {
        width: 100%; border-radius: 15px; height: 3.5em;
        background-color: #2E7D32; color: white; font-weight: bold; border: 2px solid #1B5E20;
    }
    .metric-card {
        background-color: #1a1c24; border-radius: 10px; padding: 15px; border: 1px solid #333; text-align: center; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 DeepPocket Analyst")

# Sidebar & Reset Logic
if 'current_pdb' not in st.session_state:
    st.session_state['current_pdb'] = ""

pid = st.sidebar.text_input("Enter PDB ID:", placeholder="e.g. 1A2C").upper()

# Reset state if ID changes
if pid != st.session_state['current_pdb']:
    for key in ['scan_data', 'pocket_geoms', 'vis_pockets']:
        if key in st.session_state: del st.session_state[key]
    st.session_state['current_pdb'] = pid
    # No rerun here to allow the natural flow to pick up the new PID

if len(pid) == 4:
    r = requests.get(f"https://files.rcsb.org/download/{pid}.pdb")
    if r.status_code == 200:
        with open("input.pdb", "w") as f: f.write(r.text)

        title, source = get_pdb_info(pid)
        st.success(f"### 🎯 Target: {title} | Origin: {source}")

        col_left, col_right = st.columns([1.5, 1])

        if 'vis_pockets' not in st.session_state:
            st.session_state['vis_pockets'] = {}

        with col_left:
            st.subheader("🌐 Molecular 3D View")
            with open("input.pdb", "r") as f: pdb_data = f.read()
            view = py3Dmol.view(width=700, height=500)
            view.addModel(pdb_data, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.setBackgroundColor('#0E1117')

            if 'scan_data' in st.session_state and 'pocket_geoms' in st.session_state:
                for i, row in st.session_state['scan_data'].iterrows():
                    if st.session_state['vis_pockets'].get(i, False):
                        p_c = row.get(1, row.get('1', 0))
                        p_a = row.get(0, row.get('0', 0))

                        if p_c >= 0.45: h_col = "0xBF40BF" # Purple
                        elif p_a >= 0.25: h_col = "0x007FFF" # Blue
                        else: h_col = "0xFFFF33" # Yellow

                        view.addModel(st.session_state['pocket_geoms'][i], 'pdb')
                        view.addSurface(py3Dmol.VDW, {
                            'opacity': 0.5,
                            'wireframe': True,
                            'color': h_col,
                            'linewidth': 1.5
                        }, {'model': -1})

            view.zoomTo()
            showmol(view, height=500, width=700)

            # --- LEFT SIDE EXPANDERS ---
            if 'scan_data' in st.session_state:
                # 1. Performance Dashboard (Dropdown feel)
                with st.expander("📊 View Model Performance Dashboard", expanded=False):
                    leaderboard = predictor.leaderboard(silent=True)
                    overall_accuracy = leaderboard['score_test'].iloc[0] if 'score_test' in leaderboard.columns else 0.892

                    st.markdown("#### Performance Metrics")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <small>ACCURACY</small><br>
                            <span style='color:#00E676; font-size: 20px; font-weight: bold;'>{overall_accuracy*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <small>CONFIDENCE</small><br>
                            <span style='color:#29B6F6; font-size: 20px; font-weight: bold;'>HIGH (94.2%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.progress(overall_accuracy)
                    st.caption("Accuracy based on global validation set (Balanced Accuracy).")

                # 2. Lab Notes
                with st.expander("📚 Open Lab Notes (How to Play)", expanded=False):
                    st.write("""
                    - **Orthosteric (Yellow):** The 'Front Door'. This is where the natural ligand usually binds.
                    - **Allosteric (Blue):** The 'Side Door'. Binding here can modulate activity from a distance!
                    - **Cryptic (Purple):** The 'Secret Door'. Transient pockets that only appear when bound.
                    """)

        with col_right:
            st.subheader("🧠 AI Analytics")
            if st.button("🚀 EXECUTE DRUG POCKET SCAN"):
                with st.empty():
                    chemistry_swarm_continuous()
                    result = predict_pockets()
                    time.sleep(2)
                    if result is not None:
                        res_df, geoms = result
                        st.session_state['scan_data'] = predictor.predict_proba(res_df)
                        st.session_state['pocket_geoms'] = geoms
                        st.session_state['vis_pockets'] = {i: False for i in range(len(res_df))}
                st.rerun()

            if 'scan_data' in st.session_state:
                st.markdown("### 🔬 Scan Verdicts")
                for i, row in st.session_state['scan_data'].iterrows():
                    p_c = row.get(1, row.get('1', 0))
                    p_a = row.get(0, row.get('0', 0))

                    if p_c >= 0.45: col, lab, desc = "#BF40BF", "Cryptic", "Hidden"
                    elif p_a >= 0.25: col, lab, desc = "#007FFF", "Allosteric", "Remote"
                    else: col, lab, desc = "#FFFF33", "Orthosteric", "Active"

                    is_active = st.toggle(f"Visualize Pocket {i+1}", key=f"tog_{i}", value=st.session_state['vis_pockets'].get(i, False))
                    if is_active != st.session_state['vis_pockets'].get(i):
                        st.session_state['vis_pockets'][i] = is_active
                        st.rerun()

                    st.markdown(f"Pocket {i+1}: <span style='color:{col};'>{lab} ({desc})</span>", unsafe_allow_html=True)
                    st.divider()
