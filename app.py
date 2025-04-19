"""
Tathya Fact‑Checker – Enhanced Streamlit dark‑mode interface
===========================================================
This single file contains **all** configuration, styling, and UI logic — no
`.streamlit/config.toml` needed. Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit requests pillow streamlit-lottie
"""
from __future__ import annotations

import os, json, colorsys, textwrap, requests, time
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie  # animated loaders

# ────────────────────────────
# Configuration (inline) 🛠️
# ────────────────────────────
API_ENDPOINT: str = os.getenv("TATHYA_API", "http://127.0.0.1:8000/check")
LOGO_PATH: str = os.getenv("TATHYA_LOGO", "Logo.png")
PRIMARY_COLOR = "#4D96FF"  # bright blue accent
BG_COLOR = "#0E1117"
BG_SECONDARY = "#1B1E24"
TEXT_COLOR = "#FAFAFA"
FONT_FAMILY = "Inter, sans-serif"
LOADER_URL = "https://assets5.lottiefiles.com/private_files/lf30_editor_46utqktq.json"  # free spinner
MAX_SUMMARY_WORDS = 150

# ────────────────────────────
# Helper functions
# ────────────────────────────

def hls_to_hex(hue: float, light: float = 0.5, sat: float = 0.8) -> str:
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def verdict_color(score: float) -> str:
    """Return RED→GREEN gradient colour for the given confidence 0‑1."""
    score = max(0.0, min(1.0, score))
    return hls_to_hex((120 * score) / 360)

def source_color(score: float) -> str:
    """Maps source confidence score (0.0 to 1.0) to a hex color (Red to Yellow to Green)."""
    score = max(0.0, min(1.0, score))
    return hls_to_hex((120 * score) / 360, light=0.6, sat=0.7)

@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> dict | None:
    """Fetch a Lottie animation and cache it."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def truncate(text: str, words: int = MAX_SUMMARY_WORDS) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " …"

def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        st.info("No sources returned by the API.")
        return

    # First, normalize all sources to dictionaries
    normalized_sources = []
    for src in sources:
        if isinstance(src, dict):
            # Source is already a dictionary
            normalized_sources.append(src)
        elif isinstance(src, str):
            # Source is a URL string, convert to dictionary
            # Extract domain as tool name
            tool_name = "Web Source"
            try:
                if src.startswith("http"):
                    from urllib.parse import urlparse
                    domain = urlparse(src).netloc
                    if domain:
                        tool_name = domain.replace("www.", "")
            except:
                pass
                
            normalized_sources.append({
                "url": src,
                "title": src,
                "snippet": "No preview available",
                "confidence": 0.5,
                "tool": tool_name
            })
        else:
            st.warning(f"Skipping source of unexpected type: {type(src)}")

    # No valid sources after normalization
    if not normalized_sources:
        st.warning("No valid sources found in the data.")
        return

    # Create tool badges at the top
    tools = sorted({src.get("tool", "Unknown") for src in normalized_sources})
    
    if tools:
        st.markdown("### Source Tools Used")
        with st.container():
            tool_cols = st.columns(min(len(tools), 4))  # Max 4 tools per row for better spacing
            for i, tool in enumerate(tools):
                col_index = i % 4
                tool_cols[col_index].markdown(
                    f"""<div style='background:{PRIMARY_COLOR}33;padding:8px 12px;
                    border-radius:6px;font-weight:600;text-align:center;font-size:0.9em;
                    margin-bottom:10px;border:1px solid {PRIMARY_COLOR}55;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);'>{tool}</div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("""<div style='height:15px'></div>""", unsafe_allow_html=True)

    # Display each source
    for src in normalized_sources:
        confidence = float(src.get("confidence", 0.5))
        c = source_color(confidence)
        title = src.get("title") or src.get("url", "No title")
        snippet = src.get("snippet", "No preview available")
        tool = src.get("tool", "Web Source")
        
        st.markdown(
            f"""
            <div style='border-left:6px solid {c};padding:15px;margin:15px 0;
                border-radius:8px;background:{BG_SECONDARY};position:relative;
                box-shadow:0 4px 6px rgba(0, 0, 0, 0.2);'>
                <span style='position:absolute;top:10px;right:10px;background:{PRIMARY_COLOR};color:white;
                    padding:4px 10px;border-radius:4px;font-size:12px;font-weight:bold;'>{tool}</span>
                <a href='{src.get("url")}' target='_blank' 
                    style='color:{PRIMARY_COLOR};font-weight:600;
                    text-decoration:none;font-size:16px;display:block;margin-top:5px;margin-bottom:10px;'>{title}</a>
                <span style='color:#CCCCCC;font-size:14px;'>{snippet}</span>
                <div style='clear:both;'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ────────────────────────────
# Global page settings
# ────────────────────────────
st.set_page_config(
    page_title="Tathya Fact Checker",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Tathya - A Fact Checking System developed to verify claims using multiple sources."
    }
)

# ────────────────────────────
# Custom CSS styling
# ────────────────────────────
st.markdown(
    f"""
    <style>
    /* Root variables for theming */
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --text-color: {TEXT_COLOR};
        --background-color: {BG_COLOR};
        --secondary-background-color: {BG_SECONDARY};
        --font: {FONT_FAMILY};
    }}
    
    /* Base styling */
    html, body, [class*="st"] {{ 
        font-family: var(--font);
    }}
    
    .stApp {{ 
        background-color: var(--background-color); 
        color: var(--text-color);
    }}
    
    a {{ 
        color: var(--primary-color);
    }}
    
    /* Logo size adjustment */
    .logo-container img {{
        width: auto !important;
        height: 80px !important;
    }}
    
    /* Search bar styling */
    .stTextInput > div > div > input {{
        text-align: center;
        font-size: 1.25em;
        background-color: {BG_SECONDARY};
        color: white;
        border-radius: 25px;
        border: 1px solid {PRIMARY_COLOR};
        padding: 12px 20px;
    }}
    
    /* Main container */
    .main-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}
    
    /* Logo container */
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        margin-top: 30px;
    }}
    
    /* Verdict styling */
    .verdict-container {{
        margin-top: 20px;
        margin-bottom: 25px;
    }}
    
    .verdict-text {{
        font-size: 1.8em;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}
    
    /* Summary box */
    .summary-box {{
        background-color: {BG_SECONDARY};
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        line-height: 1.6;
    }}
    
    /* Source container */
    .source-container {{
        background-color: {BG_SECONDARY};
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    
    /* Loading steps */
    .loader-step {{
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 12px 15px;
        background-color: {BG_SECONDARY};
        border-radius: 6px;
        transition: all 0.3s ease;
    }}
    
    .loader-step-active {{
        border-left: 3px solid {PRIMARY_COLOR};
        box-shadow: 0 0 8px {PRIMARY_COLOR}40;
    }}
    
    .loader-step-complete {{
        border-left: 3px solid #00CC66;
        box-shadow: 0 0 8px #00CC6640;
    }}
    
    .loader-icon {{
        margin-right: 15px;
        font-size: 18px;
    }}
    
    /* Additional tweaks */
    h1, h2, h3 {{
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }}
    
    .stApp > header {{
        background-color: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────
# Main app layout
# ────────────────────────────
# Header with logo
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
if os.path.exists(LOGO_PATH):
    logo_image = Image.open(LOGO_PATH)
    st.image(logo_image, width=350, output_format="PNG", use_container_width=False, caption="")
else:
    st.markdown("<h1 style='text-align:center;margin-bottom:0'>Tathya</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#888;margin-top:4px'>Fact‑Checking simplified</h4>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Search bar with enhanced styling
claim = st.text_input(
    "Claim to verify",
    placeholder="Enter a claim to fact check...",
    label_visibility="collapsed"
)

# Initialize session state for tracking progress
if 'progress_state' not in st.session_state:
    st.session_state.progress_state = {
        'searching': False,
        'analyzing': False,
        'verifying': False,
        'complete': False
    }

# Results container
results_container = st.container()

if claim:
    with results_container:
        # Show animated steps when not complete
        if not st.session_state.progress_state['complete']:
            # Step 1: Searching for information
            st.markdown('<div style="margin: 30px 0;">', unsafe_allow_html=True)
            
            step1_class = "loader-step loader-step-active" if not st.session_state.progress_state['searching'] else "loader-step loader-step-complete"
            st.markdown(f'<div class="{step1_class}"><span class="loader-icon">🔍</span> Searching for information related to your claim...</div>', unsafe_allow_html=True)
            if not st.session_state.progress_state['searching']:
                st.session_state.progress_state['searching'] = True
                time.sleep(0.7)  # Simulate search time
                st.rerun()
            
            # Step 2: Analyzing sources
            if st.session_state.progress_state['searching']:
                step2_class = "loader-step loader-step-active" if not st.session_state.progress_state['analyzing'] else "loader-step loader-step-complete"
                st.markdown(f'<div class="{step2_class}"><span class="loader-icon">📊</span> Analyzing sources and evidence...</div>', unsafe_allow_html=True)
                if not st.session_state.progress_state['analyzing']:
                    st.session_state.progress_state['analyzing'] = True
                    time.sleep(0.8)  # Simulate analysis time
                    st.rerun()
            
            # Step 3: Verifying claim
            if st.session_state.progress_state['analyzing']:
                step3_class = "loader-step loader-step-active" if not st.session_state.progress_state['verifying'] else "loader-step loader-step-complete"
                st.markdown(f'<div class="{step3_class}"><span class="loader-icon">✅</span> Verifying claim accuracy...</div>', unsafe_allow_html=True)
                if not st.session_state.progress_state['verifying']:
                    st.session_state.progress_state['verifying'] = True
                    time.sleep(0.9)  # Simulate verification time
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # If all steps complete, make the API call
            if st.session_state.progress_state['verifying'] and not st.session_state.progress_state['complete']:
                try:
                    # Make API call with a loading animation
                    with st.status("Contacting API...") as status:
                        lottie_json = load_lottie(LOADER_URL)
                        if lottie_json:
                            st_lottie(lottie_json, height=120, key="loading_api")
                        
                        response = requests.post(API_ENDPOINT, json={"claim": claim, "language": "en"}, timeout=60)
                        response.raise_for_status()
                        
                        api_data = response.json()
                        st.session_state.api_data = api_data
                        status.update(label="Done!", state="complete")
                    
                    # Mark process as complete
                    st.session_state.progress_state['complete'] = True
                    
                    # Brief pause before showing results
                    time.sleep(0.5)
                    st.rerun()
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error contacting the fact-checking API: {e}")
                    st.error(f"Please ensure the API server is running at {API_ENDPOINT}")
                    
                    # For demo/development - mock data if API fails
                    st.warning("Using mock data for demonstration")
                    mock_data = {
                        "result": "True",
                        "confidence_score": 0.82,
                        "explanation": "In July 2023, India surpassed China to become the world's most populous nation, driven by its sustained rapid growth and China's slowing demographic expansion. This demographic transition carries both opportunities and responsibilities for India and the wider world. Current Population Ranking: As of mid-2023, the United Nations estimates that India's population reached 1.4286 billion, slightly exceeding China's 1.4257 billion. This milestone marks the first time in recorded history that India has held the top spot. The shift reflects China's decades of low fertility and an aging population, juxtaposed with India's higher birth rates and younger demographic profile. Factors Behind India's Growth: India's population expansion stems from fertility rates at or just above replacement level, declining mortality, and improvements in healthcare.",
                        "sources": [
                            {
                                "url": "https://www.un.org/development/desa/pd/content/World-Population-Prospects-2022",
                                "title": "UN World Population Prospects 2022",
                                "snippet": "India's population reached 1.4286 billion in mid-2023, surpassing China as the world's most populous country.",
                                "confidence": 0.95,
                                "tool": "United Nations"
                            },
                            {
                                "url": "https://www.bbc.com/news/world-asia-india-65343809",
                                "title": "India overtakes China to become world's most populous nation",
                                "snippet": "India now has more people than any other country in the world, overtaking China, according to United Nations data.",
                                "confidence": 0.88,
                                "tool": "BBC News"
                            },
                            {
                                "url": "https://www.census.gov/popclock/world",
                                "title": "World Population Clock",
                                "snippet": "Current world population statistics showing India with the largest population as of 2023.",
                                "confidence": 0.75,
                                "tool": "US Census Bureau"
                            }
                        ]
                    }
                    st.session_state.api_data = mock_data
                    st.session_state.progress_state['complete'] = True
                    time.sleep(0.5)
                    st.rerun()
        
        # Display results if process is complete
        if st.session_state.progress_state['complete'] and hasattr(st.session_state, 'api_data'):
            api_data = st.session_state.api_data
            
            # --- Display Verdict ---
            verdict = api_data.get("result", "Uncertain")
            confidence = float(api_data.get("confidence_score", 0.0))
            color = verdict_color(confidence)
            
            st.markdown('<div class="verdict-container">', unsafe_allow_html=True)
            verdict_html = f"""
            <div class="verdict-text" style="background-color: {color}; color: #FFFFFF; box-shadow: 0 0 20px {color}80;">
                VERDICT: {verdict.upper()} (Confidence: {confidence:.2f})
            </div>
            """
            st.markdown(verdict_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- Display Summary ---
            with st.expander("Summary", expanded=True):
                summary = api_data.get("explanation", "No summary provided.")
                summary_truncated = truncate(summary)
                st.markdown(f"<div class='summary-box'>{summary_truncated}</div>", unsafe_allow_html=True)
                
                # Add a "Read more" button if the summary was truncated
                if len(summary.split()) > MAX_SUMMARY_WORDS:
                    with st.expander("Read full explanation"):
                        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
            
            # --- Display Sources ---
            st.subheader("Sources")
            render_sources(api_data.get("sources", []))
            
            # Reset progress state if user wants to search again
            if st.button("New Search", use_container_width=True, type="primary"):
                st.session_state.progress_state = {
                    'searching': False,
                    'analyzing': False,
                    'verifying': False,
                    'complete': False
                }
                st.rerun()
else:
    # Show prompt if no claim is entered
    with results_container:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #AAAAAA;">
            <h3>Enter a claim above to start the fact-checking process</h3>
            <p>Example: "Does India have the largest population?" or "Is climate change real?"</p>
        </div>
        """, unsafe_allow_html=True)

# Add dark/light theme toggle (visual only, as we're using dark mode by default)
with st.sidebar:
    st.title("Theme Settings")
    st.write("The app is optimized for dark mode, but you can customize your experience.")
    
    if st.toggle("Use Light Mode Theme (Visual Only)", value=False, key="light_mode"):
        st.markdown("""
        <style>
        :root {
            --background-color: #FFFFFF;
            --secondary-background-color: #F0F2F6;
            --text-color: #31333F;
        }
        </style>
        """, unsafe_allow_html=True)
        st.info("This is a visual change only. For best results, restart the app with light mode config.")

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close main container
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #888; font-size: 0.8em;">
    <p>Tathya Fact-Checking System • Powered by AI</p>
</div>
""", unsafe_allow_html=True)
