# app/streamlit_app.py
from __future__ import annotations
from pathlib import Path
import sys
import time
import streamlit as st
import os
from dotenv import load_dotenv
import joblib

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

try:
    from src.parsing.ml.skill_matcher import analyze_resume, load_skill_dataset, MODEL_PATH, VECT_PATH
    from app.auth import get_supabase_client, init_session_state, login_page, logout
    from app.metrics import calculate_metrics, create_metrics_dashboard, create_skill_comparison_chart, save_metrics_to_db, interpret_confidence, calculate_model_performance  # ← ADD THESE
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

UPLOADS_DIR = PROJ_ROOT / "app" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Resume Analyzer", page_icon="🎯", layout="wide")

init_session_state()

if st.session_state.user is None:
    login_page()
    st.stop()

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
.stApp {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
.skill-badge {display: inline-block; background: #667eea; color: white; padding: 0.5rem 1rem; margin: 0.25rem; border-radius: 20px; font-size: 0.9rem;}
.missing-badge {background: #f5576c;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### Welcome, {st.session_state.user.email}")
    if st.button("Logout"):
        logout()

    st.divider()
    st.header("Model Status")
    has_model = MODEL_PATH.exists() and VECT_PATH.exists()
    if has_model:
        st.success("✅ Trained ML Model Active")
        st.caption("Using Logistic Regression + TF-IDF")
    else:
        st.warning("⚠️ Rule-based Matching")
        st.caption("ML model not found")

    st.header("Job Roles")
    roles = load_skill_dataset()
    st.caption(f"{len(roles)} roles available")
    with st.expander("View all roles"):
        for role in sorted(roles.keys()):
            st.write(f"- {role}")

    # 🔍 DEBUG SECTION - ADD THIS
    st.markdown("---")
    st.subheader("🔍 DEBUG: Job Roles Analysis")

    # Check what's actually loaded
    st.write(f"**Total Roles Loaded:** {len(roles)}")

    # Show all roles with their skills
    for role_name, skills in roles.items():
        with st.expander(f"📊 {role_name} ({len(skills)} skills)"):
            st.write("**Skills:**", ", ".join(skills))

    # Check the source of the data
    st.write("**All Role Names:**", list(roles.keys()))

    # Check if it's the default dataset
    default_roles = ['Junior Data Scientist', 'Data Analyst']
    if set(roles.keys()) == set(default_roles):
        st.error("❌ **ISSUE:** Loading DEFAULT roles only (2 roles)")
        st.info("The function is falling back to default dataset instead of extended roles")
    else:
        st.success("✅ **SUCCESS:** Loading extended roles")

    st.markdown("---")
    # END DEBUG SECTION

    st.caption(f"{len(roles)} roles available")
    with st.expander("View all roles (Collapsed View)"):
        for role in sorted(roles.keys()):
            st.write(f"- {role}")

st.title("🎯 AI Resume Analyzer")
st.caption("Upload your resume for intelligent skill matching and gap analysis")

uploaded = st.file_uploader("📁 Upload your resume", type=["pdf", "docx"])

if uploaded:
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_name = uploaded.name.replace(" ", "_")
    saved_path = UPLOADS_DIR / f"{ts}__{safe_name}"
    with saved_path.open("wb") as f:
        f.write(uploaded.getbuffer())

    supabase = get_supabase_client()

    with st.spinner("🔍 Analyzing your resume..."):
        roles_map = load_skill_dataset()
        result = analyze_resume(str(saved_path))
        # Show ML model info
        if result.get("using_ml_model"):
            st.success("🤖 Using AI Model for Analysis")
            st.caption(f"Model Confidence: {result.get('ml_confidence', 0):.1f}%")
        else:
            st.info("⚙️ Using Rule-Based Analysis")
        # REPLACE WITH THIS UPDATED VERSION:
    try:
    # Convert user ID to string to avoid any type issues
        user_id_str = str(st.session_state.user.id)
        
        resume_record = supabase.table('resumes').insert({
            'user_id': user_id_str,  # Now using string version
            'file_name': uploaded.name,
            'parsed_skills': result["parsed"].get("skills", []),
            'parsed_education': result["parsed"].get("education", []),
            'parsed_experience': result["parsed"].get("experience", [])
        }).execute()
    
        if resume_record.data:
            resume_id = resume_record.data[0]['id']
            st.success("✅ Resume saved successfully!")
        else:
            st.error("❌ Failed to save resume - no data returned")
            resume_id = None
        
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        # Continue with the analysis even if save fails
        resume_id = None

    st.success(f"✅ Analysis complete for {uploaded.name}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Extracted Information")
        skills = result["parsed"].get("skills", [])
        edu = result["parsed"].get("education", [])
        exp = result["parsed"].get("experience", [])

        with st.container():
            st.markdown("**Skills Detected:**")
            if skills:
                skills_html = "".join([f'<span class="skill-badge">{skill.title()}</span>' for skill in sorted(skills)[:15]])
                st.markdown(skills_html, unsafe_allow_html=True)
                if len(skills) > 15:
                    st.caption(f"...and {len(skills) - 15} more")
            else:
                st.info("No skills detected")

        st.markdown("---")

        with st.expander("📚 Education", expanded=False):
            if edu:
                for e in edu[:5]:
                    st.write(f"• {e.title()}")
            else:
                st.write("Not detected")

        with st.expander("💼 Experience", expanded=False):
            if exp:
                for e in exp[:5]:
                    st.write(f"• {e.title()}")
            else:
                st.write("Not detected")

    with col2:
        st.markdown("### 🎯 Role Predictions")
        preds = result.get("predictions", [])
        if preds:
            for i, (role, score) in enumerate(preds, 1):
                st.markdown(f"**{i}.** {role} — `{score:.1f}%`")
            default_role = preds[0][0]
        else:
            default_role = result["chosen_role"]

        # 🔍 PREDICTION CONFIDENCE SECTION
        st.markdown("---")
        st.markdown("### 🔍 Prediction Confidence")

        confidence = result.get('ml_confidence', 0)
        interpretation = interpret_confidence(confidence)

        if confidence >= 60:
            st.success(f"**{interpretation}**")
        elif confidence >= 40:
            st.warning(f"**{interpretation}**")
        else:
            st.error(f"**{interpretation}**")

        st.write(f"Model Confidence Score: **{confidence:.1f}%**")

        # Add this after the role predictions section
        st.markdown("---")
        st.markdown("### 📊 Model Performance Overview")

        # Calculate overall model metrics
        try:
            if MODEL_PATH.exists() and VECT_PATH.exists():
                clf = joblib.load(MODEL_PATH)
                vect = joblib.load(VECT_PATH)
                
                model_metrics = calculate_model_performance(clf, vect)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Estimated Accuracy", f"{model_metrics['estimated_accuracy']}%")
                
                with col2:
                    st.metric("Precision", f"{model_metrics['precision_macro']}%")
                
                with col3:
                    st.metric("Recall", f"{model_metrics['recall_macro']}%")
                
                with col4:
                    st.metric("F1-Score", f"{model_metrics['f1_macro']}%")
                
                # Quality indicator
                if model_metrics['estimated_accuracy'] > 80:
                    st.success("✅ Model Quality: HIGH")
                elif model_metrics['estimated_accuracy'] > 60:
                    st.warning("⚠️ Model Quality: MEDIUM")
                else:
                    st.error("❌ Model Quality: LOW")
                    
                st.info(f"💡 {model_metrics['recommendation']}")
                
        except Exception as e:
            st.error(f"Could not load model metrics: {e}")
        st.markdown("---")

        chosen = st.selectbox(
            "🎯 Select target role for detailed analysis:",
            options=list(roles_map.keys()),
            index=list(roles_map.keys()).index(default_role) if default_role in roles_map else 0
        )

        if chosen != result["chosen_role"]:
            with st.spinner("Recalculating..."):
                result = analyze_resume(str(saved_path), chosen_role=chosen)

        matched = result["gap"].get("matched", [])
        missing = result["gap"].get("missing", [])
        match_score = result.get('match_score', 0)

        metrics = calculate_metrics(matched, missing, len(roles_map.get(chosen, [])), confidence=0.85)

        if resume_id:
            save_metrics_to_db(supabase, resume_id, chosen, match_score, matched, missing, metrics)

        st.markdown("### 📊 Performance Metrics")
        create_metrics_dashboard(metrics, match_score)

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Matched Skills**")
            if matched:
                matched_html = "".join([f'<span class="skill-badge">{s.title()}</span>' for s in sorted(matched)[:10]])
                st.markdown(matched_html, unsafe_allow_html=True)
                if len(matched) > 10:
                    st.caption(f"+{len(matched) - 10} more")
            else:
                st.info("None")

        with col_b:
            st.markdown("**❌ Missing Skills**")
            if missing:
                missing_html = "".join([f'<span class="skill-badge missing-badge">{s.title()}</span>' for s in sorted(missing)[:10]])
                st.markdown(missing_html, unsafe_allow_html=True)
                if len(missing) > 10:
                    st.caption(f"+{len(missing) - 10} more")
            else:
                st.success("None - Great match!")

        st.markdown("---")
        fig = create_skill_comparison_chart(matched, missing)
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Overall Match Score", f"{match_score:.1f}%", delta=f"{match_score - 70:.1f}%" if match_score >= 70 else None)

    if saved_path.exists():
        saved_path.unlink()

st.markdown("---")
st.markdown("### 📈 Your Analysis History")
if st.session_state.user:
    try:
        history = supabase.from_('analysis_results').select('*, resumes(file_name, upload_date)').eq('resumes.user_id', st.session_state.user.id).order('created_at', desc=True).limit(5).execute()

        if history.data:
            for record in history.data:
                with st.expander(f"📄 {record['resumes']['file_name']} - {record['predicted_role']} ({record['match_score']}%)"):
                    st.write(f"**Match Score:** {record['match_score']}%")
                    st.write(f"**Precision:** {record['precision_score']}%")
                    st.write(f"**Recall:** {record['recall_score']}%")
                    st.write(f"**F1-Score:** {record['f1_score']}%")
        else:
            st.info("No analysis history yet. Upload a resume to get started!")
    except:
        st.info("No analysis history yet. Upload a resume to get started!")

